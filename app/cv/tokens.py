import os
import math
import numpy as np
import cv2
import pytesseract
from scipy.optimize import linear_sum_assignment
from .utils import rgb_to_hsv, clamp

TOKEN_VALUES = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

MODE_TOKEN_POOLS = {
    "four": [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11],
    "six": [2, 5, 4, 6, 3, 9, 8, 11, 11, 10, 6, 3, 8, 4, 8, 10, 10, 9, 12, 12, 5, 4, 9, 5, 6, 3, 11, 2]
}

def extract_digit_roi(img_rgb):
    """
    Crops the upper/middle 40x40 region of a 56x56 token patch
    to isolate number digits while excluding bottom probability dots.
    """
    h, w, _ = img_rgb.shape
    # Center x: [8:48], Upper y: [4:44]
    roi = img_rgb[4:44, 8:48]
    if roi.size == 0:
        return np.zeros((40, 40, 3), dtype=np.uint8)
    return cv2.resize(roi, (40, 40), interpolation=cv2.INTER_LINEAR)

def is_double_digit(digit_patch):
    """
    Determines if a 40x40 digit patch contains a double digit (10, 11, 12) vs single digit (2..9).
    Uses color-aware text ink binarization and connected component topology:
    - Double digits (10, 11, 12) consist of 2 distinct side-by-side digit components.
    - Single digits (2..9) consist of 1 connected component.
    """
    inner = digit_patch[2:26, 4:36]
    if inner.size == 0:
        return False
        
    luma = inner[:, :, 0] * 0.299 + inner[:, :, 1] * 0.587 + inner[:, :, 2] * 0.114
    hsv = cv2.cvtColor(inner, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    h_chan = hsv[:, :, 0]
    
    # Isolate dark or red digit text ink from background
    dark = luma < 140
    red = (luma < 175) & (sat > 0.28) & ((h_chan < 20) | (h_chan > 160))
    ink = (dark | red).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink)
    
    significant_x = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 12:
            significant_x.append(centroids[i][0])
            
    if len(significant_x) >= 2:
        if (max(significant_x) - min(significant_x)) >= 7.0:
            return True
            
    return False

EXPECTED_TOKEN_PIPS = {2: 1, 12: 1, 3: 2, 11: 2, 4: 3, 10: 3, 5: 4, 9: 4, 6: 5, 8: 5}

def count_dots_from_width(w: int, area: int = 0) -> int:
    if area >= 70 and w >= 28:
        return 4
    if w <= 8:
        return 1
    elif w <= 19 and area <= 34:
        return 2
    elif w <= 27:
        return 3
    elif w <= 34:
        return 4
    else:
        return 5

def detect_probability_pips(patch, is_red=False):
    """
    Counts probability dots/pips (1..5) in the lower band of a 56x56 token patch.
    Uses quality-scored multi-window sliding search (y: 32..44) with color-aware binarization to compensate for crop offsets.
    Groups candidate components by horizontal row alignment (delta cy <= 2.5px) to reject out-of-row noise and digit curves.
    """
    best_cnt = None
    best_score = -1
    
    for y_start in (32, 34, 36, 38, 40, 42, 44):
        band = patch[y_start:min(56, y_start+12), 8:48]
        if band.size == 0:
            continue
            
        luma = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0
        h_chan = hsv[:, :, 0]
        
        beige_mask = (luma > 165) & (sat < 0.30)
        if np.sum(beige_mask) > 10:
            bg_luma = np.median(luma[beige_mask])
        else:
            bg_luma = np.median(luma)
            
        dark = (luma < bg_luma - 38) & (luma < 140)
        red = (luma < 210) & (sat > 0.20) & ((h_chan < 25) | (h_chan > 155))
        ink = red.astype(np.uint8) if is_red else dark.astype(np.uint8)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink)
        
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cy_c = centroids[i][1]
            if cy_c >= 0.8 and 4 <= area <= 75 and w <= 30 and 2 <= h <= 8:
                candidates.append((area, w, h, cy_c))
                
        if not candidates:
            continue
            
        # Group candidates by Y-coordinate row alignment (within 2.5px)
        median_cy = np.median([c[3] for c in candidates])
        row_comps = [c for c in candidates if abs(c[3] - median_cy) <= 2.5]
        if not row_comps:
            continue
            
        pips_cnt = sum(count_dots_from_width(c[1], c[0]) for c in row_comps)
        # Quality score measures component sharpness (thin clean dots h<=4 over blurred bands h>=5)
        mean_h = np.mean([c[2] for c in row_comps])
        quality_score = 100.0 - mean_h * 15.0 - abs(median_cy - 2.5) * 5.0
            
        if 1 <= pips_cnt <= 5:
            if quality_score > best_score:
                best_score = quality_score
                best_cnt = pips_cnt
                
    if best_cnt is not None:
        if not is_red and best_cnt == 5:
            best_cnt = 4
    return best_cnt



_numeric_template_masks = None

def ensure_numeric_template_masks():
    global _numeric_template_masks
    if _numeric_template_masks is not None:
        return _numeric_template_masks
        
    store = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    templates_dir = os.path.join(base_dir, "templates")
    
    for val in TOKEN_VALUES:
        path = os.path.join(templates_dir, f"num_{val}.png")
        if os.path.exists(path):
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                roi = extract_digit_roi(cv2.resize(img_rgb, (56, 56)))
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                bg_luma = np.median(gray)
                _, thresh = cv2.threshold(gray, max(110, bg_luma - 25), 255, cv2.THRESH_BINARY_INV)
                store[val] = (thresh > 0).astype(np.float32)
    _numeric_template_masks = store
    return _numeric_template_masks

def shift_image(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def predict_numeric_ocr_scores(digit_roi):
    """
    Numeric OCR Engine: Binarizes unknown digit glyph and matches against 
    canonical Catan numeric font templates via shift-invariant IoU.
    """
    tmpl_masks = ensure_numeric_template_masks()
    gray = cv2.cvtColor(digit_roi, cv2.COLOR_RGB2GRAY) if len(digit_roi.shape) == 3 else digit_roi.copy()
    bg_luma = np.median(gray)
    _, thresh = cv2.threshold(gray, max(110, bg_luma - 25), 255, cv2.THRESH_BINARY_INV)
    unknown = (thresh > 0).astype(np.float32)
    
    scores = {}
    for val in TOKEN_VALUES:
        tmpl = tmpl_masks.get(val)
        if tmpl is None or tmpl.shape != unknown.shape:
            scores[val] = 0.0
            continue
        best_iou = 0.0
        for dx in (-3, -1, 0, 1, 3):
            for dy in (-3, -1, 0, 1, 3):
                s_unk = shift_image(unknown, dx, dy)
                inter = np.sum(s_unk * tmpl)
                union = np.sum(s_unk) + np.sum(tmpl) - inter
                iou = float(inter / max(1.0, union))
                if iou > best_iou:
                    best_iou = iou
        scores[val] = best_iou
    return scores

def run_tesseract_ocr(digit_patch):
    """
    Pre-processes digit patch (binarization, padding, upscaling)
    and executes Tesseract OCR for digit detection.
    """
    try:
        gray = cv2.cvtColor(digit_patch, cv2.COLOR_RGB2GRAY)
        # Resize to 120x120 for crisp OCR reading
        resized = cv2.resize(gray, (120, 120), interpolation=cv2.INTER_CUBIC)
        # Otsu thresholding
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Add white border margin
        padded = cv2.copyMakeBorder(thresh, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        
        text = pytesseract.image_to_string(
            padded,
            config='--psm 7 -c tessedit_char_whitelist=0123456789'
        ).strip()
        
        if text.isdigit():
            val = int(text)
            if val in TOKEN_VALUES:
                return val
    except Exception:
        pass
    return None

def compute_ssim(img1, img2):
    """
    Computes Structural Similarity Index (SSIM) between two grayscale images.
    """
    g1 = (img1[:, :, 0] * 0.299 + img1[:, :, 1] * 0.587 + img1[:, :, 2] * 0.114).astype(np.float32)
    g2 = (img2[:, :, 0] * 0.299 + img2[:, :, 1] * 0.587 + img2[:, :, 2] * 0.114).astype(np.float32)
    
    k1, k2 = 0.01, 0.03
    L = 255.0
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))




def normalize_vector(v):
    mag = float(np.linalg.norm(v))
    if mag == 0.0:
        return np.zeros_like(v, dtype=np.float32)
    return (v / mag).astype(np.float32)

def extract_edge_vector(img_rgb, out_width, out_height):
    resized = cv2.resize(img_rgb, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Zero out outer border to match Sobel interior loop
    gx[0, :] = 0.0
    gx[-1, :] = 0.0
    gx[:, 0] = 0.0
    gx[:, -1] = 0.0
    gy[0, :] = 0.0
    gy[-1, :] = 0.0
    gy[:, 0] = 0.0
    gy[:, -1] = 0.0
    
    edge = np.sqrt(gx * gx + gy * gy)
    return normalize_vector(edge.flatten())

def build_ink_mask(img_rgb):
    resized = img_rgb.astype(np.float32)
    luma = resized[:, :, 0] * 0.299 + resized[:, :, 1] * 0.587 + resized[:, :, 2] * 0.114
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    
    dark_ink = np.clip((170.0 - luma) / 170.0, 0.0, 1.0)
    sat_ink = np.clip((sat - 0.18) * 1.2, 0.0, 1.0)
    
    mask = dark_ink * 0.72 + sat_ink * 0.28
    return normalize_vector(mask.flatten())

def make_variants(img_rgb, out_width, out_height, scales, offsets):
    variants = []
    for scale in scales:
        for offset in offsets:
            # Create blank black image
            variant = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            
            draw_w = int(out_width * scale)
            draw_h = int(out_height * scale)
            
            # Resize source template
            resized = cv2.resize(img_rgb, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
            
            # Compute paste offsets
            dx = int((out_width - draw_w) / 2 + offset["dx"])
            dy = int((out_height - draw_h) / 2 + offset["dy"])
            
            # Safe bounding for pasting
            src_x0 = max(0, -dx)
            src_y0 = max(0, -dy)
            dst_x0 = max(0, dx)
            dst_y0 = max(0, dy)
            
            src_w = min(draw_w - src_x0, out_width - dst_x0)
            src_h = min(draw_h - src_y0, out_height - dst_y0)
            
            if src_w > 0 and src_h > 0:
                variant[dst_y0:dst_y0+src_h, dst_x0:dst_x0+src_w] = resized[src_y0:src_y0+src_h, src_x0:src_x0+src_w]
                
            variants.append(variant)
    return variants

def cosine_similarity(a, b):
    return float(np.sum(a * b))

def best_similarity(vector, bank):
    if isinstance(bank, np.ndarray):
        return float(np.max(np.dot(bank, vector)))
    elif isinstance(bank, list) and len(bank) > 0:
        bank_mat = np.array(bank, dtype=np.float32)
        return float(np.max(np.dot(bank_mat, vector)))
    return float('-inf')

def calibrate_canvas_colors(img_rgb):
    mean_r = np.mean(img_rgb[:, :, 0])
    mean_g = np.mean(img_rgb[:, :, 1])
    mean_b = np.mean(img_rgb[:, :, 2])
    gray = (mean_r + mean_g + mean_b) / 3.0
    scale_r = gray / max(1.0, mean_r)
    scale_g = gray / max(1.0, mean_g)
    scale_b = gray / max(1.0, mean_b)
    
    calibrated = np.zeros_like(img_rgb, dtype=np.uint8)
    for c in range(3):
        scale = scale_r if c == 0 else (scale_g if c == 1 else scale_b)
        scaled_ch = (img_rgb[:, :, c].astype(np.float32) * scale) / 255.0
        calibrated[:, :, c] = np.clip(np.power(scaled_ch, 0.95) * 255.0, 0, 255).astype(np.uint8)
        
    return calibrated

# Template store singleton representation in Python
_token_template_store = None

def ensure_token_template_store():
    global _token_template_store
    if _token_template_store is not None:
        return _token_template_store
        
    store = {}
    scales = [0.9, 1.0, 1.1]
    offsets = [
        {"dx": 0, "dy": 0},
        {"dx": -1, "dy": 0},
        {"dx": 1, "dy": 0},
        {"dx": 0, "dy": -1},
        {"dx": 0, "dy": 1}
    ]
    
    # Path relative to working directory or absolute
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    templates_path = os.path.join(base_dir, "templates")
    
    for val in TOKEN_VALUES:
        filename = f"num_{val}.png"
        path = os.path.join(templates_path, filename)
        
        # Read image
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load token template at path: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Base resize to 56x56
        base = cv2.resize(img_rgb, (56, 56), interpolation=cv2.INTER_LINEAR)
        variants = make_variants(base, 56, 56, scales, offsets)
        
        store[val] = {
            "edgeVectors": np.array([extract_edge_vector(v, 56, 56) for v in variants], dtype=np.float32),
            "inkVectors": np.array([build_ink_mask(v) for v in variants], dtype=np.float32),
            "digitROIs": [extract_digit_roi(v) for v in variants]
        }
        
    _token_template_store = store
    return _token_template_store

def global_assign_tokens(scored_land_tiles, mode_key):
    token_pool = MODE_TOKEN_POOLS.get(mode_key) or MODE_TOKEN_POOLS["four"]
    if len(scored_land_tiles) != len(token_pool):
        return scored_land_tiles
        
    max_score = float('-inf')
    score_matrix = np.zeros((len(scored_land_tiles), len(token_pool)), dtype=np.float32)
    
    for r_idx, entry in enumerate(scored_land_tiles):
        for c_idx, token_val in enumerate(token_pool):
            score = entry["scores"].get(token_val, -8.0)
            score_matrix[r_idx, c_idx] = score
            if score > max_score:
                max_score = score
                
    if not math.isfinite(max_score):
        max_score = 0.0
        
    cost_matrix = max_score - score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assigned_entries = []
    for r_idx in range(len(scored_land_tiles)):
        entry = scored_land_tiles[r_idx]
        assigned_token = token_pool[col_ind[r_idx]]
        assigned_score = entry["scores"].get(assigned_token, float('-inf'))
        
        second_best_score = float('-inf')
        for val in TOKEN_VALUES:
            if val != assigned_token:
                second_best_score = max(second_best_score, entry["scores"].get(val, float('-inf')))
                
        confidence = clamp(0.5 + (assigned_score - second_best_score) * 0.25, 0.0, 1.0)
        
        entry_copy = dict(entry)
        entry_copy["token"] = assigned_token
        entry_copy["tokenConfidence"] = confidence
        assigned_entries.append(entry_copy)
        
    return assigned_entries

def detect_tokens(center_result, tile_result):
    templates = ensure_token_template_store()
    mode_key = "six" if center_result["modeKey"] == "six" else "four"
    
    calibrated_img = calibrate_canvas_colors(center_result["normalizedImage"])
    h_img, w_img, _ = calibrated_img.shape
    
    hex_w = center_result["geometry"]["hexW"]
    hex_h = center_result["geometry"]["hexH"]
    search_dist = hex_w * 0.25
    
    land_tiles = [tile for tile in tile_result["tiles"] if tile["resource"] != "desert"]
    
    scored_land_tiles = []
    for tile in land_tiles:
        tile_center = next((c for c in center_result["centers"] if c["tileId"] == tile["tileId"]), None)
        cx = tile_center["x"]
        cy = tile_center["y"]
        
        # Token markers on Colonist.io web canvas rendered at lower portion of hex tile
        target_cx = cx
        target_cy = cy + hex_h * 0.20
        
        # Find closest valid token marker (rejecting Ore rocks at upper hex)
        closest_marker = None
        min_dist = float('inf')
        markers = center_result["markerDebug"]["markers"]
        for m in markers:
            d = math.hypot(m["x"] - target_cx, m["y"] - target_cy)
            if d < search_dist and d < min_dist:
                min_dist = d
                closest_marker = m
                
        if closest_marker:
            cx = closest_marker["x"]
            cy = closest_marker["y"]
        else:
            cx = target_cx
            cy = target_cy
            
        span_w = hex_w * 0.32
        span_h = hex_h * 0.32
        
        # Crop 56x56 patch around token center
        x0_i = max(0, int(round(cx - span_w / 2.0)))
        y0_i = max(0, int(round(cy - span_h / 2.0)))
        x1_i = min(w_img, int(round(cx + span_w / 2.0)))
        y1_i = min(h_img, int(round(cy + span_h / 2.0)))
        
        cropped = calibrated_img[y0_i:y1_i, x0_i:x1_i]
        if cropped.size == 0:
            patch = np.zeros((56, 56, 3), dtype=np.uint8)
            raw_patch = np.zeros((56, 56, 3), dtype=np.uint8)
        else:
            patch = cv2.resize(cropped, (56, 56), interpolation=cv2.INTER_LINEAR)
            raw_cropped = center_result["normalizedImage"][y0_i:y1_i, x0_i:x1_i]
            raw_patch = cv2.resize(raw_cropped, (56, 56), interpolation=cv2.INTER_LINEAR)
            
        digit_roi = extract_digit_roi(patch)
        numeric_ocr_scores = predict_numeric_ocr_scores(digit_roi)
        has_double_width = is_double_digit(digit_roi)
        
        # Red token heuristic (evaluate on bounded digit_roi)
        hsv_roi = cv2.cvtColor(digit_roi, cv2.COLOR_RGB2HSV)
        h_chan, s_chan, v_chan = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]
        r_mask = ((h_chan < 15) | (h_chan > 165)) & (s_chan > 40) & (v_chan > 70)
        red_count = np.sum(r_mask)
        is_red = red_count >= 15
        
        scores = {}
        best_token = None
        best_score = float('-inf')
        second_best_score = float('-inf')
        
        for val in TOKEN_VALUES:
            t = templates[val]
            ocr_s = numeric_ocr_scores.get(val, 0.0)
            ssim_score = max(0.0, max(compute_ssim(digit_roi, tmpl_roi) for tmpl_roi in t["digitROIs"]))
            
            # Numeric OCR + SSIM Template Ensemble
            score = ocr_s * 0.50 + ssim_score * 0.50
            
            # Strict domain rules: Red ink (6/8) & Double digit width (10/11/12)
            if val == 6 or val == 8:
                score += 0.5 if is_red else -2.0
            else:
                score += -2.0 if is_red else 0.0
                
            if val in (10, 11, 12):
                score += 0.5 if has_double_width else -2.0
            else:
                score += -2.0 if has_double_width else 0.10
                
            scores[val] = score
            
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_token = val
            elif score > second_best_score:
                second_best_score = score
                
        confidence = clamp(0.5 + (best_score - second_best_score) * 0.25, 0.0, 1.0)
        
        scored_land_tiles.append({
            "tileId": tile["tileId"],
            "scores": scores,
            "x": cx,
            "y": cy,
            "token": best_token,
            "tokenConfidence": confidence,
            "isDoubleWidth": has_double_width
        })
        
    globally_assigned = global_assign_tokens(scored_land_tiles, mode_key)
    
    all_tiles = []
    for tile in tile_result["tiles"]:
        if tile["resource"] == "desert":
            all_tiles.append({
                **tile,
                "token": None,
                "tokenConfidence": 1.0,
                "isDoubleWidth": False,
                "pipCount": None
            })
        else:
            assigned = next((t for t in globally_assigned if t["tileId"] == tile["tileId"]), None)
            all_tiles.append({
                **tile,
                "token": assigned["token"] if assigned else None,
                "tokenConfidence": assigned["tokenConfidence"] if assigned else 0.0,
                "isDoubleWidth": assigned["isDoubleWidth"] if assigned else False
            })
            
    return {
        "modeKey": mode_key,
        "tiles": all_tiles
    }
