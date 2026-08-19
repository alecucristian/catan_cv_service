import os
import math
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from .utils import clamp
from .tokens import extract_edge_vector, build_ink_mask, best_similarity

HARBOR_TYPES = ["3:1", "wood 2:1", "brick 2:1", "sheep 2:1", "wheat 2:1", "ore 2:1"]

MODE_HARBOR_POOLS = {
    "four": ["3:1", "3:1", "3:1", "3:1", "wood 2:1", "brick 2:1", "sheep 2:1", "wheat 2:1", "ore 2:1"],
    "six": ["3:1", "3:1", "3:1", "3:1", "3:1", "3:1", "wood 2:1", "brick 2:1", "sheep 2:1", "wheat 2:1", "ore 2:1"]
}

MODE_FRAME_SLOTS = {
    "four": 18,
    "six": 22
}

def make_harbor_variants(img_rgb, out_width, out_height, scales, angles, offsets):
    variants = []
    w, h = out_width, out_height
    for scale in scales:
        for angle in angles:
            for offset in offsets:
                dx = offset["dx"]
                dy = offset["dy"]
                
                # Canvas-equivalent affine warp math
                cos_t = math.cos(angle)
                sin_t = math.sin(angle)
                
                a = scale * cos_t
                b = -scale * sin_t
                c = scale * sin_t
                d = scale * cos_t
                
                tx = w/2.0 + a * (-w/2.0 + dx) + b * (-h/2.0 + dy)
                ty = h/2.0 + c * (-w/2.0 + dx) + d * (-h/2.0 + dy)
                
                M = np.array([
                    [a, b, tx],
                    [c, d, ty]
                ], dtype=np.float32)
                
                warped = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                variants.append(warped)
    return variants

_harbor_template_store = None

def ensure_harbor_template_store():
    global _harbor_template_store
    if _harbor_template_store is not None:
        return _harbor_template_store
        
    store = {}
    angles = [0, math.pi / 3.0, 2 * math.pi / 3.0, math.pi, 4 * math.pi / 3.0, 5 * math.pi / 3.0]
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
    
    # Harbor template file names mapping
    template_files = {
        "3:1": "harbor_3to1.png",
        "wood 2:1": "harbor_wood.png",
        "brick 2:1": "harbor_brick.png",
        "sheep 2:1": "harbor_sheep.png",
        "wheat 2:1": "harbor_grain.png",
        "ore 2:1": "harbor_ore.png"
    }
    
    for harbor_type in HARBOR_TYPES:
        filename = template_files[harbor_type]
        path = os.path.join(templates_path, filename)
        
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load harbor template at path: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        base = cv2.resize(img_rgb, (56, 56), interpolation=cv2.INTER_LINEAR)
        variants = make_harbor_variants(base, 56, 56, scales, angles, offsets)
        
        store[harbor_type] = {
            "edgeVectors": np.array([extract_edge_vector(v, 56, 56) for v in variants], dtype=np.float32),
            "inkVectors": np.array([build_ink_mask(v) for v in variants], dtype=np.float32)
        }
        
    _harbor_template_store = store
    return _harbor_template_store

def global_assign_harbors(scored_slots, mode_key):
    harbors_pool = list(MODE_HARBOR_POOLS.get(mode_key) or MODE_HARBOR_POOLS["four"])
    
    # Filter scored_slots to ONLY include even slotIndexes (0, 2, 4, 6, 8, 10, 12, 14, 16)
    even_slots = [s for s in scored_slots if s["slotIndex"] % 2 == 0]
    
    total_even_slots = len(even_slots)
    empty_count = total_even_slots - len(harbors_pool)
    for _ in range(empty_count):
        harbors_pool.append("empty")
        
    if len(even_slots) != len(harbors_pool):
        return scored_slots
        
    empty_threshold = 0.53
    max_score = float('-inf')
    score_matrix = np.zeros((len(even_slots), len(harbors_pool)), dtype=np.float32)
    
    for r_idx, entry in enumerate(even_slots):
        for c_idx, label in enumerate(harbors_pool):
            if label == "empty":
                score = empty_threshold
            else:
                score = entry["scores"].get(label, -8.0)
            score_matrix[r_idx, c_idx] = score
            if score > max_score:
                max_score = score
                
    if not math.isfinite(max_score):
        max_score = 0.0
        
    cost_matrix = max_score - score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    even_assigned_map = {}
    for r_idx in range(len(even_slots)):
        slot_idx = even_slots[r_idx]["slotIndex"]
        assigned_label = harbors_pool[col_ind[r_idx]]
        even_assigned_map[slot_idx] = assigned_label
        
    assigned_slots = []
    for entry in scored_slots:
        entry_copy = dict(entry)
        entry_copy["assigned"] = even_assigned_map.get(entry["slotIndex"], "empty")
        assigned_slots.append(entry_copy)
        
    return assigned_slots

def detect_harbors(center_result):
    templates = ensure_harbor_template_store()
    mode_key = "six" if center_result["modeKey"] == "six" else "four"
    
    from .tokens import calibrate_canvas_colors
    calibrated_img = calibrate_canvas_colors(center_result["normalizedImage"])
    h_img, w_img, _ = calibrated_img.shape
    
    hex_w = center_result["geometry"]["hexW"]
    frame_slots = center_result["frameSlots"]
    
    # Load raw RGB harbor templates (40x40)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tmpl_files = {
        "3:1": "templates/harbor_3to1.png",
        "wood 2:1": "templates/harbor_wood.png",
        "brick 2:1": "templates/harbor_brick.png",
        "sheep 2:1": "templates/harbor_sheep.png",
        "wheat 2:1": "templates/harbor_grain.png",
        "ore 2:1": "templates/harbor_ore.png"
    }
    tmpl_rgb = {}
    for htype, path_rel in tmpl_files.items():
        path = os.path.join(base_dir, path_rel)
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        tmpl_rgb[htype] = cv2.resize(rgb, (40, 40))
        
    scored_slots = []
    gray_img = cv2.cvtColor(calibrated_img, cv2.COLOR_RGB2GRAY)
    
    for slot in frame_slots:
        sx, sy = int(round(slot["x"])), int(round(slot["y"]))
        # Search radius around island proximity point to snap onto white sail
        R = int(round(hex_w * 0.32))
        y0_s, y1_s = max(0, sy - R), min(h_img, sy + R)
        x0_s, x1_s = max(0, sx - R), min(w_img, sx + R)
        search_gray = gray_img[y0_s:y1_s, x0_s:x1_s]
        
        snap_x, snap_y = sx, sy
        if search_gray.size > 0:
            mask_sail = search_gray > 200
            if np.any(mask_sail):
                ys_s, xs_s = np.where(mask_sail)
                snap_y = int(round(y0_s + np.mean(ys_s)))
                snap_x = int(round(x0_s + np.mean(xs_s)))
                
        span = int(round(hex_w * 0.45))
        y0 = max(0, snap_y - span // 2)
        y1 = min(h_img, snap_y + span // 2)
        x0 = max(0, snap_x - span // 2)
        x1 = min(w_img, snap_x + span // 2)
        cropped = calibrated_img[y0:y1, x0:x1]
        
        if cropped.size == 0:
            flag_crop = np.zeros((40, 40, 3), dtype=np.uint8)
        else:
            gray_c = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            mask = gray_c > 180
            if np.any(mask):
                ys, xs = np.where(mask)
                cy, cx = int(round(np.mean(ys))), int(round(np.mean(xs)))
                r_crop = 18
                cy0, cy1 = max(0, cy - r_crop), min(cropped.shape[0], cy + r_crop)
                cx0, cx1 = max(0, cx - r_crop), min(cropped.shape[1], cx + r_crop)
                patch = cropped[cy0:cy1, cx0:cx1]
                flag_crop = cv2.resize(patch, (40, 40)) if patch.size > 0 else cv2.resize(cropped, (40, 40))
            else:
                flag_crop = cv2.resize(cropped, (40, 40))
                
        feat = flag_crop.astype(np.float32) / 255.0
        scores = {}
        for htype in HARBOR_TYPES:
            t_img = tmpl_rgb[htype]
            t_feat = t_img.astype(np.float32) / 255.0
            best_diff = 1.0
            for dx in (-2, 0, 2):
                for dy in (-2, 0, 2):
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    s_feat = cv2.warpAffine(feat, M, (40, 40))
                    diff = float(np.mean(np.abs(s_feat - t_feat)))
                    if diff < best_diff: best_diff = diff
            scores[htype] = 1.0 - best_diff
            
        scored_slots.append({
            "slotIndex": slot["slotIndex"],
            "x": slot["x"],
            "y": slot["y"],
            "angle": slot["angle"],
            "scores": scores
        })
        
    globally_assigned = global_assign_harbors(scored_slots, mode_key)
    
    ports = []
    for slot in globally_assigned:
        assigned = slot["assigned"]
        if assigned != "empty":
            ports.append({
                "slotIndex": slot["slotIndex"],
                "type": assigned,
                "x": slot["x"],
                "y": slot["y"]
            })
            
    return {
        "modeKey": mode_key,
        "ports": ports
    }

