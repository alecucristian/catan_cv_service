import os
import math
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from .utils import rgb_to_hsv, clamp

TILE_CONFIDENCE_THRESHOLD = 0.58

MODE_RESOURCE_COUNTS = {
    "four": {"wood": 4, "brick": 3, "sheep": 4, "wheat": 4, "ore": 3, "desert": 1},
    "six": {"wood": 6, "brick": 5, "sheep": 6, "wheat": 6, "ore": 5, "desert": 2}
}

RESOURCE_OPTIONS = ["wood", "brick", "sheep", "wheat", "ore", "desert"]

_resource_template_store = None

def ensure_resource_template_store():
    global _resource_template_store
    if _resource_template_store is not None:
        return _resource_template_store
        
    store = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    templates_path = os.path.join(base_dir, "templates")
    
    template_files = {
        "wood": "res_wood.png",
        "brick": "res_brick.png",
        "sheep": "res_sheep.png",
        "wheat": "res_grain.png",
        "ore": "res_ore.png",
        "desert": "res_desert.png"
    }
    
    for resource, filename in template_files.items():
        path = os.path.join(templates_path, filename)
        img_rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_rgba is not None and img_rgba.shape[2] == 4:
            # Resize template to 26x26 for icon patch matching
            tmpl_resized = cv2.resize(img_rgba, (26, 26), interpolation=cv2.INTER_AREA)
            tmpl_bgr = tmpl_resized[:, :, :3]
            tmpl_rgb = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2RGB)
            tmpl_mask = tmpl_resized[:, :, 3]
            store[resource] = {
                "rgb": tmpl_rgb,
                "mask": tmpl_mask
            }
            
    _resource_template_store = store
    return _resource_template_store

def extract_tile_icon_patch(img_rgb, center, geometry):
    h_img, w_img, _ = img_rgb.shape
    icon_cx = center["x"]
    icon_cy = center["y"] - geometry["hexH"] * 0.21
    
    patch_w = geometry["hexW"] * 0.35
    patch_h = geometry["hexH"] * 0.35
    
    x0 = max(0, int(round(icon_cx - patch_w / 2.0)))
    y0 = max(0, int(round(icon_cy - patch_h / 2.0)))
    x1 = min(w_img, int(round(icon_cx + patch_w / 2.0)))
    y1 = min(h_img, int(round(icon_cy + patch_h / 2.0)))
    
    cropped = img_rgb[y0:y1, x0:x1]
    if cropped.size == 0:
        return np.zeros((36, 36, 3), dtype=np.uint8)
        
    return cv2.resize(cropped, (36, 36), interpolation=cv2.INTER_LINEAR)


def extract_tile_body_features(img_rgb, center, geometry):
    """
    Extracts body color and texture features from a tile center.
    Matches the JS implementation in stage-tiles.js exactly.
    """
    h_img, w_img, _ = img_rgb.shape
    span_w = geometry["hexW"] * 0.74
    span_h = geometry["hexH"] * 0.74
    
    # Calculate crop coordinates
    x0 = center["x"] - span_w / 2.0
    y0 = center["y"] - span_h / 2.0
    
    # Crop and resize to 72x72
    # Ensure safe bounding
    x0_i = max(0, int(round(x0)))
    y0_i = max(0, int(round(y0)))
    x1_i = min(w_img, int(round(x0 + span_w)))
    y1_i = min(h_img, int(round(y0 + span_h)))
    
    cropped = img_rgb[y0_i:y1_i, x0_i:x1_i]
    if cropped.size == 0:
        return {"h": 0.0, "s": 0.0, "v": 0.0, "texture": 0.0}
        
    patch = cv2.resize(cropped, (72, 72), interpolation=cv2.INTER_LINEAR)
    
    # Calculate luma
    luma = np.zeros((72, 72), dtype=np.float32)
    for y in range(72):
        for x in range(72):
            color = patch[y, x]
            luma[y, x] = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114) / 255.0
            
    sum_h = 0.0
    sum_s = 0.0
    sum_v = 0.0
    count = 0
    texture_energy = 0.0
    
    for y in range(2, 70):
        for x in range(2, 70):
            nx = ((x + 0.5) / 72.0) * 2.0 - 1.0
            ny = ((y + 0.5) / 72.0) * 2.0 - 1.0
            
            # Hex mask
            if (abs(nx) + abs(ny) * 0.85) > 0.92:
                continue
                
            # Token center mask
            if nx*nx + ny*ny < 0.36:
                continue
                
            # Icon mask
            icon_dx = nx
            icon_dy = ny + 0.21
            if icon_dx*icon_dx + icon_dy*icon_dy < 0.055:
                continue
                
            color = patch[y, x]
            h, s, v = rgb_to_hsv(color[0], color[1], color[2])
            
            sum_h += h
            sum_s += s
            sum_v += v
            count += 1
            
            gx = luma[y, x + 1] - luma[y, x - 1]
            gy = luma[y + 1, x] - luma[y - 1, x]
            texture_energy += math.sqrt(gx*gx + gy*gy)
            
    if count == 0:
        return {"h": 0.0, "s": 0.0, "v": 0.0, "texture": 0.0}
        
    return {
        "h": sum_h / count,
        "s": sum_s / count,
        "v": sum_v / count,
        "texture": texture_energy / count
    }

def score_body_features(features, resource):
    h = features["h"]
    s = features["s"]
    v = features["v"]
    
    score = 0.0
    if resource == "brick":
        if h < 35.0 or h > 340.0:
            score += 2.0
        elif h < 45.0:
            score += 1.0
        else:
            score -= 2.0
        if s > 0.45:
            score += 1.0
        else:
            score -= 1.0
            
    elif resource == "wheat":
        if 30.0 <= h <= 65.0:
            score += 2.0
        elif h > 20.0 and h < 75.0:
            score += 1.0
        else:
            score -= 2.0
        if s > 0.45:
            score += 1.0
        else:
            score -= 1.0
            
    elif resource == "sheep":
        if 55.0 <= h <= 125.0:
            score += 2.0
        elif h > 45.0 and h < 140.0:
            score += 1.0
        else:
            score -= 2.0
        if s > 0.4:
            score += 1.0
        else:
            score -= 1.0
        if v > 0.5:
            score += 0.5
            
    elif resource == "wood":
        if 90.0 <= h <= 165.0:
            score += 2.0
        elif h > 75.0 and h < 180.0:
            score += 1.0
        else:
            score -= 2.0
        if v < 0.75:
            score += 1.0
        else:
            score -= 1.0
            
    elif resource == "ore":
        if s < 0.38:
            score += 3.0
        elif s < 0.45:
            score += 1.0
        else:
            score -= 3.0
            
    elif resource == "desert":
        if 35.0 <= h <= 75.0:
            score += 1.0
        else:
            score -= 1.0
        if 0.25 < s < 0.55:
            score += 2.0
        else:
            score -= 1.0
            
    return score

def build_repeated_label_pool(target_counts, ordered_labels):
    pool = []
    for label in ordered_labels:
        count = target_counts.get(label, 0)
        for _ in range(count):
            pool.append(label)
    return pool

def z_score_normalize(ranked_scores):
    values = [x["score"] for x in ranked_scores]
    mean = sum(values) / float(max(1, len(values)))
    variance = sum((x - mean) ** 2 for x in values) / float(max(1, len(values) - 1))
    std = math.sqrt(max(variance, 1e-8))
    
    normalized = []
    for x in ranked_scores:
        item = dict(x)
        item["score"] = (x["score"] - mean) / std
        normalized.append(item)
    return normalized

def score_tile_resources(img_rgb, center, geometry):
    body_features = extract_tile_body_features(img_rgb, center, geometry)
    templates = ensure_resource_template_store()
    icon_patch = extract_tile_icon_patch(img_rgb, center, geometry)
    
    score_by_resource = {}
    template_weight = 0.40
    
    for resource in RESOURCE_OPTIONS:
        body_score = score_body_features(body_features, resource)
        
        # Template match score
        tmpl_score = 0.0
        if resource in templates:
            tmpl = templates[resource]
            try:
                res_map = cv2.matchTemplate(icon_patch, tmpl["rgb"], cv2.TM_CCOEFF_NORMED, mask=tmpl["mask"])
                _, max_val, _, _ = cv2.minMaxLoc(res_map)
                tmpl_score = max(0.0, float(max_val))
            except Exception:
                tmpl_score = 0.0
                
        combined = (1.0 - template_weight) * body_score + template_weight * tmpl_score * 3.0
        score_by_resource[resource] = combined
        
    ranked = []
    for resource in RESOURCE_OPTIONS:
        ranked.append({
            "resource": resource,
            "score": score_by_resource[resource],
            "bodyScore": score_body_features(body_features, resource)
        })
        
    normalized = z_score_normalize(ranked)
    normalized.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "ranked": normalized,
        "bodyFeatures": body_features
    }

def global_assign_resources(entries, mode_key):
    target_counts = MODE_RESOURCE_COUNTS.get(mode_key) or MODE_RESOURCE_COUNTS["four"]
    label_pool = build_repeated_label_pool(target_counts, RESOURCE_OPTIONS)
    
    if len(entries) != len(label_pool):
        return entries
        
    max_score = float('-inf')
    score_matrix = np.zeros((len(entries), len(label_pool)), dtype=np.float32)
    
    for r_idx, entry in enumerate(entries):
        for c_idx, label in enumerate(label_pool):
            found = next((x for x in entry["scores"] if x["resource"] == label), None)
            score = found["score"] if found else -8.0
            score_matrix[r_idx, c_idx] = score
            if score > max_score:
                max_score = score
                
    if not math.isfinite(max_score):
        max_score = 0.0
        
    cost_matrix = max_score - score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assigned_entries = []
    for r_idx in range(len(entries)):
        assigned_label = label_pool[col_ind[r_idx]]
        entry = entries[r_idx]
        
        # Check top choice resource
        sorted_scores = sorted(entry["scores"], key=lambda x: x["score"], reverse=True)
        if sorted_scores:
            top_res = sorted_scores[0]["resource"]
            top_score = sorted_scores[0]["score"]
            second_score = sorted_scores[1]["score"] if len(sorted_scores) > 1 else top_score - 1.0
            
            # If top choice is very confident, honor top choice to handle custom scenario boards
            if top_score > 0.85 and (top_score - second_score) > 0.15:
                assigned_label = top_res
                
        assigned_score = next((x["score"] for x in entry["scores"] if x["resource"] == assigned_label), float('-inf'))
        next_score = next((x["score"] for x in entry["scores"] if x["resource"] != assigned_label), assigned_score - 1.0)
        
        entry_copy = dict(entry)
        entry_copy["resource"] = assigned_label
        entry_copy["confidence"] = clamp(0.44 + (assigned_score - next_score) * 0.2 + max(0.0, assigned_score) * 0.08, 0.0, 1.0)
        assigned_entries.append(entry_copy)
        
    return assigned_entries

def detect_tiles(center_result):
    mode_key = "six" if center_result["modeKey"] == "six" else "four"
    img_rgb = center_result["normalizedImage"]
    
    scored = []
    for tile_center in center_result["centers"]:
        scored_tile = score_tile_resources(img_rgb, tile_center, center_result["geometry"])
        scored.append({
            **tile_center,
            "scores": scored_tile["ranked"],
            "bodyFeatures": scored_tile["bodyFeatures"]
        })
        
    assigned = global_assign_resources(scored, mode_key)
    
    final_tiles = []
    for tile in assigned:
        final_tiles.append({
            "tileId": tile["tileId"],
            "row": tile["row"],
            "col": tile["col"],
            "resource": tile.get("resource") or tile["scores"][0]["resource"],
            "confidence": tile.get("confidence") or clamp(0.45 + (tile["scores"][0]["score"] - (tile["scores"][1]["score"] if len(tile["scores"]) > 1 else tile["scores"][0]["score"] - 1.0)) * 0.2, 0.0, 1.0),
            "colorImportance": 1.0,
            "iconImportance": 0.0,
            "bodyFeatures": tile["bodyFeatures"],
            "alternatives": tile["scores"][:3]
        })
        
    low_confidence_count = sum(1 for t in final_tiles if t["confidence"] < TILE_CONFIDENCE_THRESHOLD)
    average_confidence = sum(t["confidence"] for t in final_tiles) / float(max(1, len(final_tiles)))
    
    return {
        "modeKey": mode_key,
        "tiles": final_tiles,
        "quality": {
            "lowConfidenceCount": low_confidence_count,
            "averageConfidence": average_confidence,
            "averageColorImportance": 1.0,
            "threshold": TILE_CONFIDENCE_THRESHOLD
        }
    }
