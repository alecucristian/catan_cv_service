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
    total_slots = MODE_FRAME_SLOTS.get(mode_key) or MODE_FRAME_SLOTS["four"]
    
    empty_count = total_slots - len(harbors_pool)
    for _ in range(empty_count):
        harbors_pool.append("empty")
        
    if len(scored_slots) != len(harbors_pool):
        return scored_slots
        
    empty_threshold = 0.53
    max_score = float('-inf')
    score_matrix = np.zeros((len(scored_slots), len(harbors_pool)), dtype=np.float32)
    
    for r_idx, entry in enumerate(scored_slots):
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
    
    assigned_slots = []
    for r_idx in range(len(scored_slots)):
        entry = scored_slots[r_idx]
        assigned_label = harbors_pool[col_ind[r_idx]]
        
        entry_copy = dict(entry)
        entry_copy["assigned"] = assigned_label
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
    
    scored_slots = []
    for slot in frame_slots:
        span_w = hex_w * 0.34
        span_h = hex_w * 0.34
        
        # Crop patch around slot
        x0_i = max(0, int(round(slot["x"] - span_w / 2.0)))
        y0_i = max(0, int(round(slot["y"] - span_h / 2.0)))
        x1_i = min(w_img, int(round(slot["x"] + span_w / 2.0)))
        y1_i = min(h_img, int(round(slot["y"] + span_h / 2.0)))
        
        cropped = calibrated_img[y0_i:y1_i, x0_i:x1_i]
        if cropped.size == 0:
            patch = np.zeros((56, 56, 3), dtype=np.uint8)
        else:
            patch = cv2.resize(cropped, (56, 56), interpolation=cv2.INTER_LINEAR)
            
        edge_vector = extract_edge_vector(patch, 56, 56)
        ink_vector = build_ink_mask(patch)
        
        scores = {}
        for harbor_type in HARBOR_TYPES:
            t = templates[harbor_type]
            edge_score = best_similarity(edge_vector, t["edgeVectors"])
            ink_score = best_similarity(ink_vector, t["inkVectors"])
            
            scores[harbor_type] = ink_score * 0.74 + edge_score * 0.26
            
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

