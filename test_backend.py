import os
import sys
import glob
import re
import cv2
import numpy as np

# Add app to path so we can import app.cv
sys.path.insert(0, "/app")

from app.cv.pipeline import run_pipeline

def parse_board_code(code_str):
    """
    Parses a board code string into tile codes and port codes.
    E.g. S9O10S8...P0T2O4G -> tiles: ['S9', 'O10', 'S8', ...], harbors: ['0T', '2O', '4G', ...]
    """
    clean = code_str.replace(" ", "")
    if "P" in clean:
        tile_part, harbor_part = clean.split("P", 1)
    else:
        tile_part, harbor_part = clean, ""
        
    tiles = [f"{r}{num}" for r, num in re.findall(r'([SOGWBD])(\d*)', tile_part)]
    harbors = [f"{slot}{htype}" for slot, htype in re.findall(r'(\d+)([TWBSGO])', harbor_part)]
    return tiles, harbors

def fit_text_scale(text, font, initial_scale, max_width, min_scale=0.2, thickness=1):
    """
    Calculates font_scale and text size ensuring the text width does not exceed max_width.
    """
    scale = initial_scale
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw > max_width and tw > 0:
        scale = max(min_scale, scale * (max_width / float(tw)))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    return scale, tw, th

def draw_detection_overlay(img_bgr, result, expected_code=None, filename=""):
    """
    Draws detected hex centers, terrain types, token numbers, harbor positions,
    and a summary banner onto the image with auto-fitted text.
    """
    overlay = img_bgr.copy()
    h, w, _ = overlay.shape
    
    orig_bounds = result["bounds"]["original"]
    scale_x = orig_bounds["w"] / 960.0
    scale_y = orig_bounds["h"] / 960.0
    off_x = orig_bounds["x"]
    off_y = orig_bounds["y"]

    def to_orig(nx, ny):
        return int(round(off_x + nx * scale_x)), int(round(off_y + ny * scale_y))

    # 1. Draw bounding box around detected board area
    bx0, by0 = max(0, orig_bounds["x"]), max(0, orig_bounds["y"])
    bx1, by1 = min(w, orig_bounds["x"] + orig_bounds["w"]), min(h, orig_bounds["y"] + orig_bounds["h"])
    cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (255, 165, 0), 3)

    # Maps for lookup
    slots_by_index = {s["slotIndex"]: s for s in result.get("frameSlots", [])}
    tiles_by_id = {t["tileId"]: t for t in result.get("tiles", [])}

    # BGR Colors for resources
    RESOURCE_COLORS = {
        "wood": (34, 139, 34),       # Dark Green
        "brick": (42, 42, 165),      # Reddish Brown
        "sheep": (120, 210, 100),    # Light Green
        "wheat": (0, 215, 255),      # Gold / Yellow
        "ore": (160, 160, 160),      # Slate Gray
        "desert": (140, 190, 230)    # Tan / Sand
    }

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 2. Draw Hex Tiles
    radius = int(round(35 * scale_x))
    radius = max(22, min(65, radius))

    for center in result.get("centers", []):
        tile_id = center["tileId"]
        tile = tiles_by_id.get(tile_id, {})
        cx, cy = to_orig(center["x"], center["y"])

        resource = tile.get("resource", "desert")
        token = tile.get("token")

        color = RESOURCE_COLORS.get(resource, (200, 200, 200))

        # Filled circle for tile center
        cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.circle(overlay, (cx, cy), radius, (0, 0, 0), 2)

        # Label text (e.g. W10, G5, D, O6, B8, S11)
        res_initial = resource[0].upper() if resource != "wheat" else "G"
        token_str = "" if token is None else str(token)
        label = f"{res_initial}{token_str}"

        # Text color contrast
        text_color = (0, 0, 0) if resource in ("wheat", "sheep", "desert") else (255, 255, 255)

        init_scale = max(0.45, radius / 38.0)
        thickness = max(1, int(round(init_scale * 1.5)))
        font_scale, tw, th = fit_text_scale(label, font, init_scale, int(radius * 1.6), 0.25, thickness)

        if resource == "desert" or token is None:
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(overlay, label, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)
        else:
            # Draw main token label in upper half
            tx = cx - tw // 2
            ty = cy - 2
            cv2.putText(overlay, label, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # Build sub-label for debug inspection: e.g. "2W •3" or "1W •2"
            w_str = "2W" if tile.get("isDoubleWidth") else "1W"
            p_val = tile.get("pipCount")
            p_str = f"p{p_val}" if p_val is not None else "p-"
            sub_label = f"{w_str} {p_str}"

            sub_scale, stw, sth = fit_text_scale(sub_label, font, init_scale * 0.65, int(radius * 1.6), 0.2, 1)
            stx = cx - stw // 2
            sty = cy + sth + 4
            cv2.putText(overlay, sub_label, (stx, sty), font, sub_scale, text_color, 1, cv2.LINE_AA)

    # 3. Draw Ports / Harbors
    for port in result.get("ports", []):
        slot_index = port["slotIndex"]
        label = port["label"]  # e.g. "wood 2:1" or "3:1"
        slot = slots_by_index.get(slot_index)
        
        norm_x = port.get("x", slot["x"] if slot else 0)
        norm_y = port.get("y", slot["y"] if slot else 0)
        px, py = to_orig(norm_x, norm_y)

        p_radius = int(round(22 * scale_x))
        p_radius = max(18, min(40, p_radius))

        # Harbor badge (Cyan)
        cv2.circle(overlay, (px, py), p_radius, (255, 220, 0), -1)
        cv2.circle(overlay, (px, py), p_radius, (0, 0, 0), 2)

        # Format port label into 2 lines: Line 1 = P{slot_index}, Line 2 = Harbor Type
        res_map = {"wood": "W", "brick": "B", "sheep": "S", "wheat": "G", "ore": "O"}
        if "3:1" in label:
            line1 = f"P{slot_index}"
            line2 = "3:1"
        else:
            res_code = label.split()[0].lower()
            code_letter = res_map.get(res_code, res_code[0].upper())
            line1 = f"P{slot_index}"
            line2 = f"{code_letter} 2:1"

        max_harbor_w = int(p_radius * 1.7)

        scale1, tw1, th1 = fit_text_scale(line1, font, 0.45, max_harbor_w, 0.2, 1)
        scale2, tw2, th2 = fit_text_scale(line2, font, 0.38, max_harbor_w, 0.2, 1)

        cv2.putText(overlay, line1, (px - tw1 // 2, py - 2), font, scale1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(overlay, line2, (px - tw2 // 2, py + th2 + 2), font, scale2, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. Draw Header Banner with Expected vs Detected status
    det_code = result["boardCode"].replace(" ", "")
    det_clean = det_code
    exp_clean = expected_code.replace(" ", "") if expected_code else None

    is_match = (exp_clean == det_clean) if exp_clean else None
    
    line_count = 4 if expected_code else 3
    banner_h = max(115, line_count * 28)
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)

    alpha = 0.88
    overlay[0:banner_h, 0:w] = cv2.addWeighted(overlay[0:banner_h, 0:w], 1 - alpha, banner, alpha, 0)

    max_banner_w = w - 30

    # Line 1: Title & Status
    title_str = f"File: {filename}" if filename else "Catan CV Detection Result"
    t_scale, tw, th = fit_text_scale(title_str, font, 0.55, max_banner_w - 180, 0.3, 2)
    cv2.putText(overlay, title_str, (15, 26), font, t_scale, (255, 255, 255), 2, cv2.LINE_AA)

    if is_match is not None:
        status_str = "MATCH: PASSED" if is_match else "MATCH: FAILED"
        status_color = (0, 255, 0) if is_match else (50, 50, 255)
        s_scale, sw, sh = fit_text_scale(status_str, font, 0.55, 160, 0.3, 2)
        cv2.putText(overlay, status_str, (w - sw - 15, 26), font, s_scale, status_color, 2, cv2.LINE_AA)

    # Line 2: Detected Line
    det_str = f"Detected: {det_code}"
    d_scale, dw, dh = fit_text_scale(det_str, font, 0.45, max_banner_w, 0.2, 1)
    cv2.putText(overlay, det_str, (15, 54), font, d_scale, (0, 255, 255), 1, cv2.LINE_AA)

    # Line 3: Expected Line
    if expected_code:
        exp_str = f"Expected: {expected_code}"
        e_scale, ew, eh = fit_text_scale(exp_str, font, 0.45, max_banner_w, 0.2, 1)
        cv2.putText(overlay, exp_str, (15, 80), font, e_scale, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Line 4: Stats Line
        stats_str = f"Mode: {result['modeKey']} | Hexes: {len(result['centers'])} | Ports: {len(result['ports'])} | Quality: {result['quality']['overall']:.3f}"
        st_scale, stw, sth = fit_text_scale(stats_str, font, 0.42, max_banner_w, 0.2, 1)
        cv2.putText(overlay, stats_str, (15, 104), font, st_scale, (170, 170, 170), 1, cv2.LINE_AA)
    else:
        stats_str = f"Mode: {result['modeKey']} | Hexes: {len(result['centers'])} | Ports: {len(result['ports'])} | Quality: {result['quality']['overall']:.3f}"
        st_scale, stw, sth = fit_text_scale(stats_str, font, 0.42, max_banner_w, 0.2, 1)
        cv2.putText(overlay, stats_str, (15, 80), font, st_scale, (170, 170, 170), 1, cv2.LINE_AA)

    return overlay

def process_image(image_path, output_dir="./test_outputs"):
    filename = os.path.basename(image_path)
    expected_code = os.path.splitext(filename)[0]

    print(f"\n==========================================")
    print(f"Testing image: {filename}")
    print(f"==========================================")
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image from {image_path}!")
        return False
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    print("Running board detection pipeline...")
    result = run_pipeline(img_rgb, "four")
    
    detected_code = result['boardCode']
    det_clean = detected_code.replace(" ", "")
    exp_clean = expected_code.replace(" ", "")

    exp_tiles, exp_ports = parse_board_code(expected_code)
    det_tiles, det_ports = parse_board_code(detected_code)

    matching_tiles = sum(1 for e, d in zip(exp_tiles, det_tiles) if e == d)
    total_exp_tiles = len(exp_tiles)

    # Parse harbors for matching count
    def parse_h_map(p_list):
        h_map = {}
        for p in p_list:
            m = re.match(r'(\d+)([TWBSGO])', p)
            if m:
                h_map[int(m.group(1))] = m.group(2)
        # Check if single letter at end is slot 0
        if 0 not in h_map:
            m0 = re.search(r'([TWBSGO])$', "".join(p_list))
            if m0: h_map[0] = m0.group(1)
        return h_map

    exp_h_map = parse_h_map(exp_ports)
    det_h_map = parse_h_map(det_ports)
    matching_harbors = sum(1 for slot, htype in exp_h_map.items() if det_h_map.get(slot) == htype)
    total_exp_harbors = max(9, len(exp_h_map))

    is_match = (det_clean == exp_clean)

    print("Pipeline Execution Complete:")
    print(f"  Detected Board Code: {det_clean}")
    print(f"  Expected Board Code: {exp_clean}")
    print(f"  Matching Tiles:      {matching_tiles}/{total_exp_tiles}")
    print(f"  Matching Harbors:    {matching_harbors}/{total_exp_harbors}")
    print(f"  Overall Match:       {'PASSED' if is_match else 'FAILED'}")

    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"output_{filename}"
    output_path = os.path.join(output_dir, out_filename)

    annotated_img = draw_detection_overlay(img_bgr, result, expected_code=expected_code, filename=filename)
    cv2.imwrite(output_path, annotated_img)
    
    # Also save to ./test_output.jpeg for backward compatibility
    cv2.imwrite("./test_output.jpeg", annotated_img)
    
    print(f"  Overlay result saved to: {output_path}")

    return is_match

def main():
    test_dir = "./test_images"
    image_paths = []

    if os.path.exists(test_dir):
        for ext in ("*.jpeg", "*.jpg", "*.png", "*.webp"):
            image_paths.extend(glob.glob(os.path.join(test_dir, ext)))

    if not image_paths:
        # Fallback to catan_image.jpeg if test_images directory has no images
        fallback = "./catan_image.jpeg"
        if os.path.exists(fallback):
            image_paths = [fallback]
        else:
            print("No test images found in ./test_images or workspace root!")
            sys.exit(1)

    print(f"Found {len(image_paths)} test image(s) in test dataset.")

    passed_count = 0
    total_count = len(image_paths)

    for img_path in sorted(image_paths):
        try:
            passed = process_image(img_path)
            if passed:
                passed_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n==========================================")
    print(f"TEST SUMMARY: {passed_count}/{total_count} PASSED")
    print(f"==========================================")

if __name__ == "__main__":
    main()


