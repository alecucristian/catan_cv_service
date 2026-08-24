import os
import math
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from .utils import clamp

HARBOR_TYPES = ["3:1", "wood 2:1", "brick 2:1", "sheep 2:1", "wheat 2:1", "ore 2:1"]

MODE_HARBOR_POOLS = {
    "four": ["3:1", "3:1", "3:1", "3:1", "wood 2:1", "brick 2:1", "sheep 2:1", "wheat 2:1", "ore 2:1"],
    "six": ["3:1", "3:1", "3:1", "3:1", "3:1", "3:1", "wood 2:1", "brick 2:1", "sheep 2:1", "wheat 2:1", "ore 2:1"]
}

MODE_FRAME_SLOTS = {
    "four": 18,
    "six": 22
}

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
    mode_key = "six" if center_result["modeKey"] == "six" else "four"
    
    # Digital screenshots use pristine flat RGB signatures (no grey-world distortion)
    calibrated_img = center_result["normalizedImage"]
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
    tmpls_bgr = {}
    for htype, path_rel in tmpl_files.items():
        path = os.path.join(base_dir, path_rel)
        tmpl_img = cv2.imread(path)
        if tmpl_img is not None:
            tmpls_bgr[htype] = tmpl_img
        else:
            tmpls_bgr[htype] = np.zeros((70, 70, 3), dtype=np.uint8)
        
    scored_slots = []
    gray_img = cv2.cvtColor(calibrated_img, cv2.COLOR_RGB2GRAY)
    
    for slot in frame_slots:
        sx, sy = int(round(slot["x"])), int(round(slot["y"]))
        # Search radius around island proximity point to snap onto white sail
        R = int(round(hex_w * 0.28))
        search = gray_img[max(0, sy - R):min(h_img, sy + R), max(0, sx - R):min(w_img, sx + R)]
        snap_x, snap_y = sx, sy
        if search.size > 0 and np.any(search > 200):
            ys, xs = np.where(search > 200)
            snap_y = max(0, sy - R) + int(round(np.mean(ys)))
            snap_x = max(0, sx - R) + int(round(np.mean(xs)))
            
        crop_rgb = calibrated_img[max(0, snap_y - 36):min(h_img, snap_y + 36), max(0, snap_x - 36):min(w_img, snap_x + 36)]
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR) if crop_rgb.size > 0 else np.zeros((40, 40, 3), dtype=np.uint8)

        scores = {}
        for htype, t_img in tmpls_bgr.items():
            best_val = -1.0
            for scale in (0.36, 0.40, 0.44, 0.48, 0.52):
                tw = int(round(t_img.shape[1] * scale))
                th = int(round(t_img.shape[0] * scale))
                if crop_bgr.shape[0] >= th and crop_bgr.shape[1] >= tw:
                    t_scaled = cv2.resize(t_img, (tw, th))
                    m_res = cv2.matchTemplate(crop_bgr, t_scaled, cv2.TM_CCOEFF_NORMED)
                    val = float(np.max(m_res))
                    if val > best_val:
                        best_val = val
            scores[htype] = best_val

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

