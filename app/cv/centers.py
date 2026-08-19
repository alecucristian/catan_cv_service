import math
import numpy as np
import cv2
from typing import TypedDict, Dict, Any, List, Optional
from .utils import rgb_to_hsv, sample_hsv, clamp

class FitResult(TypedDict):
    t: Dict[str, float]
    inlierCount: int
    meanResidual: float
    score: float


OCEAN_RGB = {"r": 9, "g": 102, "b": 164}
NORMALIZED_BOARD_SIZE = 960

MODE_ROWS = {
    "four": [3, 4, 5, 4, 3],
    "six": [3, 4, 5, 6, 5, 4, 3]
}

MODE_FRAME_SLOTS = {
    "four": 18,
    "six": 22
}

CENTER_QUALITY_THRESHOLDS = {
    "markerCountWarn": 10,
    "markerCountGood": 16,
    "residualWarn": 0.16,
    "residualGood": 0.09
}

def fallback_board_bounds(w, h):
    portrait = h > w * 1.25
    if portrait:
        bw = int(w * 0.62)
        bh = int(h * 0.42)
        return {
            "x": int((w - bw) / 2),
            "y": int(h * 0.31),
            "w": bw,
            "h": bh
        }
    size = int(min(w, h) * 0.66)
    return {
        "x": int((w - size) / 2),
        "y": int((h - size) / 2),
        "w": size,
        "h": size
    }

def color_distance_sq(r, g, b, tr, tg, tb):
    dr = int(r) - int(tr)
    dg = int(g) - int(tg)
    db = int(b) - int(tb)
    return dr * dr + dg * dg + db * db

def is_ocean_like_pixel(r, g, b):
    h, s, v = rgb_to_hsv(r, g, b)
    dist_sq = color_distance_sq(r, g, b, OCEAN_RGB["r"], OCEAN_RGB["g"], OCEAN_RGB["b"])
    close_to_target = dist_sq <= 9200
    hue_band = 182 <= h <= 224 and s >= 0.28 and v >= 0.16
    return close_to_target or hue_band

def compute_ocean_ring_score(ocean_mask, w, h, min_x, min_y, max_x, max_y):
    ocean_hits = 0
    samples = 0
    pad = 3
    x0 = max(0, min_x - pad)
    y0 = max(0, min_y - pad)
    x1 = min(w - 1, max_x + pad)
    y1 = min(h - 1, max_y + pad)

    for x in range(x0, x1 + 1):
        top = y0 * w + x
        bottom = y1 * w + x
        samples += 2
        ocean_hits += 1 if ocean_mask[top] else 0
        ocean_hits += 1 if ocean_mask[bottom] else 0
        
    for y in range(y0 + 1, y1):
        left = y * w + x0
        right = y * w + x1
        samples += 2
        ocean_hits += 1 if ocean_mask[left] else 0
        ocean_hits += 1 if ocean_mask[right] else 0

    return ocean_hits / samples if samples > 0 else 0

def extract_best_island_component(mask_2d, ocean_mask_2d, w, h):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_2d, connectivity=8)
    if num_labels <= 1:
        return None
        
    diag = math.hypot(w, h)
    cx, cy = w / 2.0, h / 2.0
    best = None
    ocean_flat = ocean_mask_2d.flatten()

    for i in range(1, num_labels):
        count = int(stats[i, cv2.CC_STAT_AREA])
        if count < 80:
            continue
            
        min_x = int(stats[i, cv2.CC_STAT_LEFT])
        min_y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        max_x = min_x + bw - 1
        max_y = min_y + bh - 1
        
        box_area = bw * bh
        area_ratio = count / float(w * h)
        aspect = bw / float(max(1, bh))
        shape_score = clamp(1.0 - abs(math.log(aspect)) / 1.25, 0.0, 1.0)
        fill_score = clamp((count / float(box_area) - 0.2) / 0.6, 0.0, 1.0)
        area_score = clamp(1.0 - abs(0.2 - area_ratio) / 0.2, 0.0, 1.0)
        ccx = float(centroids[i][0])
        ccy = float(centroids[i][1])
        center_score = clamp(1.0 - math.hypot(ccx - cx, ccy - cy) / (diag * 0.72), 0.0, 1.0)
        ocean_ring_score = compute_ocean_ring_score(ocean_flat, w, h, min_x, min_y, max_x, max_y)
        touches_edge = min_x <= 1 or min_y <= 1 or max_x >= w - 2 or max_y >= h - 2
        edge_penalty = 0.58 if touches_edge else 1.0
        
        score = (count * 
                 (0.3 + 0.7 * shape_score) * 
                 (0.3 + 0.7 * fill_score) * 
                 (0.25 + 0.75 * center_score) * 
                 (0.2 + 0.8 * ocean_ring_score) * 
                 (0.25 + 0.75 * area_score) * 
                 edge_penalty)
                 
        if not best or score > best["score"]:
            best = {
                "minX": min_x, "maxX": max_x,
                "minY": min_y, "maxY": max_y,
                "count": count, "score": score,
                "oceanRingScore": ocean_ring_score,
                "areaRatio": area_ratio
            }
            
    return best

def detect_board_bounds(img_rgb):
    h, w, _ = img_rgb.shape
    sample_w = min(640, w)
    scale = w / float(sample_w)
    sample_h = max(1, int(round(h / scale)))
    
    # Resize image to 640px sample width for high precision candidate detection
    sample_img = cv2.resize(img_rgb, (sample_w, sample_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sample_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    H = hsv[:, :, 0] * 2.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0
    
    bh = (H // 10).astype(np.int64) * 10
    bs = (np.round(S * 10.0) / 10.0).astype(np.float32)
    bv = (np.round(V * 10.0) / 10.0).astype(np.float32)
    
    combined = (bh << 32) | ((bs * 100).astype(np.int64) << 16) | (bv * 100).astype(np.int64)
    vals, counts = np.unique(combined, return_counts=True)
    best_combined = vals[np.argmax(counts)]
    
    bg_h = float((best_combined >> 32) & 0xFFFF) + 5.0
    bg_s = float(((best_combined >> 16) & 0xFFFF) / 100.0) + 0.05
    bg_v = float((best_combined & 0xFFFF) / 100.0) + 0.05
    
    h_diff = np.minimum(np.abs(H - bg_h), 360.0 - np.abs(H - bg_h))
    is_bg = (h_diff < 30.0) & (np.abs(S - bg_s) < 0.4) & (np.abs(V - bg_v) < 0.4)
    
    r = sample_img[:, :, 0].astype(np.int32)
    g = sample_img[:, :, 1].astype(np.int32)
    b = sample_img[:, :, 2].astype(np.int32)
    dist_sq = (r - OCEAN_RGB["r"])**2 + (g - OCEAN_RGB["g"])**2 + (b - OCEAN_RGB["b"])**2
    close_to_target = dist_sq <= 9200
    hue_band = (H >= 182) & (H <= 224) & (S >= 0.28) & (V >= 0.16)
    is_ocean = close_to_target | hue_band
    
    ocean_mask_2d = (is_bg | is_ocean).astype(np.uint8)
    land_mask_2d = (1 - ocean_mask_2d).astype(np.uint8)
    
    component = extract_best_island_component(land_mask_2d, ocean_mask_2d, sample_w, sample_h)
    if not component or component["count"] < 800:
        return fallback_board_bounds(w, h)
        
    pad_x = (component["maxX"] - component["minX"] + 1) * 0.08
    pad_y = (component["maxY"] - component["minY"] + 1) * 0.08
    
    rx = max(0, int(round((component["minX"] - pad_x) * scale)))
    ry = max(0, int(round((component["minY"] - pad_y) * scale)))
    right = min(w, int(round((component["maxX"] + 1 + pad_x) * scale)))
    bottom = min(h, int(round((component["maxY"] + 1 + pad_y) * scale)))
    
    rw = max(1, right - rx)
    rh = max(1, bottom - ry)
    
    # Isotropic Square Expansion: Expand symmetrically to 1:1 square centered on the island
    size = int(round(max(rw, rh)))
    cx = rx + rw / 2.0
    cy = ry + rh / 2.0
    x_out = int(round(cx - size / 2.0))
    y_out = int(round(cy - size / 2.0))
    
    return {
        "x": x_out,
        "y": y_out,
        "w": size,
        "h": size
    }

def build_skeleton(rows):
    tiles = []
    tile_id = 0
    for row in range(len(rows)):
        for col in range(rows[row]):
            tiles.append({"id": tile_id, "row": row, "col": col})
            tile_id += 1
    return tiles

def tile_key(row, col):
    return f"{row}:{col}"

def build_adjacency(tiles, rows):
    by_key = {}
    for tile in tiles:
        by_key[tile_key(tile["row"], tile["col"])] = tile["id"]
        
    adjacency = {}
    for tile in tiles:
        neighbors = set()
        current = rows[tile["row"]]
        up = rows[tile["row"] - 1] if tile["row"] > 0 else None
        down = rows[tile["row"] + 1] if tile["row"] < len(rows) - 1 else None
        
        left_id = by_key.get(tile_key(tile["row"], tile["col"] - 1))
        right_id = by_key.get(tile_key(tile["row"], tile["col"] + 1))
        if left_id is not None:
            neighbors.add(left_id)
        if right_id is not None:
            neighbors.add(right_id)
            
        if up is not None:
            offsets = [-1, 0] if up == current - 1 else [0, 1]
            for o in offsets:
                n_id = by_key.get(tile_key(tile["row"] - 1, tile["col"] + o))
                if n_id is not None:
                    neighbors.add(n_id)
                    
        if down is not None:
            offsets = [-1, 0] if down == current - 1 else [0, 1]
            for o in offsets:
                n_id = by_key.get(tile_key(tile["row"] + 1, tile["col"] + o))
                if n_id is not None:
                    neighbors.add(n_id)
                    
        adjacency[tile["id"]] = list(neighbors)
    return adjacency

def build_layout(rows, bounds):
    tiles = build_skeleton(rows)
    adjacency = build_adjacency(tiles, rows)
    max_cols = max(rows)
    width_factor = max_cols * 0.88 + 2.1
    v_ratio = math.sqrt(3) / 2.0  # ~0.8660254
    height_factor = ((len(rows) - 1) * 0.88 * v_ratio) + 1.15 + 1.8
    
    hex_w = min(bounds["w"] / width_factor, bounds["h"] / height_factor)
    h_step = hex_w * 0.88
    v_step = h_step * v_ratio
    hex_h = h_step * (2.0 / math.sqrt(3.0))
    
    land_width = max_cols * h_step + hex_w * 0.2
    land_height = (len(rows) - 1) * v_step + hex_h
    pad_x = (bounds["w"] - land_width) / 2.0
    pad_y = (bounds["h"] - land_height) / 2.0
    
    centers = {}
    for tile in tiles:
        row_count = rows[tile["row"]]
        x = bounds["x"] + pad_x + ((max_cols - row_count) * h_step) / 2.0 + tile["col"] * h_step + hex_w / 2.0
        y = bounds["y"] + pad_y + tile["row"] * v_step + hex_h / 2.0
        centers[tile["id"]] = {"x": x, "y": y}
        
    return {
        "tiles": tiles,
        "adjacency": adjacency,
        "centers": centers,
        "geometry": {"hexW": hex_w, "hexH": hex_h, "hStep": h_step, "vStep": v_step, "rows": rows},
        "boardCenter": {"x": bounds["x"] + bounds["w"] / 2.0, "y": bounds["y"] + bounds["h"] / 2.0}
    }

def sort_clockwise_by_center(ids, centers, cx, cy):
    def angle_sort(tile_id):
        c = centers[tile_id]
        return math.atan2(c["y"] - cy, c["x"] - cx)
    return sorted(ids, key=angle_sort)

def build_spiral_order(tiles, adjacency, centers, board_center):
    remaining = set(tile["id"] for tile in tiles)
    rings = []
    
    while len(remaining) > 0:
        ring = []
        for tile_id in remaining:
            neighbors = adjacency[tile_id]
            inside = sum(1 for n in neighbors if n in remaining)
            if inside < 6:
                ring.append(tile_id)
                
        if len(ring) == 0:
            ring = list(remaining)
            
        ordered = sort_clockwise_by_center(ring, centers, board_center["x"], board_center["y"])
        
        # Shift to start at top-left-most tile
        def shift_sort(idx):
            c = centers[ordered[idx]]
            return (c["y"], c["x"])
            
        start_idx = sorted(range(len(ordered)), key=lambda idx: (ordered[idx] in remaining, shift_sort(idx)))[0]
        # Wait, the JS sorting logic:
        # sorted( (a,b) => { const dy = a.center.y - b.center.y; if (abs(dy) > 40) return dy; return a.center.x - b.center.x })
        # Let's recreate exactly:
        def custom_start_key(a_idx):
            c = centers[ordered[a_idx]]
            # We want to find the top-left-most
            # Let's just find the index of the element that matches JS's sort
            return c
            
        # We can implement standard JS comparison using a helper
        best_idx = 0
        best_center = centers[ordered[0]]
        for index in range(1, len(ordered)):
            c = centers[ordered[index]]
            dy = c["y"] - best_center["y"]
            if abs(dy) > 40.0:
                if dy < 0:
                    best_idx = index
                    best_center = c
            else:
                dx = c["x"] - best_center["x"]
                if dx < 0:
                    best_idx = index
                    best_center = c
                    
        shifted = ordered[best_idx:] + ordered[:best_idx]
        rings.append(shifted)
        for tile_id in shifted:
            remaining.discard(tile_id)
            
    # Flatten rings list
    return [item for sublist in rings for item in sublist]

def build_frame_slots(tiles, adjacency, centers, board_center, geometry, slot_count):
    boundary = [tile for tile in tiles if len(adjacency[tile["id"]]) < 6]
    vectors = [
        {"x": geometry["hStep"], "y": 0.0},
        {"x": -geometry["hStep"], "y": 0.0},
        {"x": geometry["hStep"] / 2.0, "y": geometry["vStep"]},
        {"x": -geometry["hStep"] / 2.0, "y": geometry["vStep"]},
        {"x": geometry["hStep"] / 2.0, "y": -geometry["vStep"]},
        {"x": -geometry["hStep"] / 2.0, "y": -geometry["vStep"]}
    ]
    
    def key_for(x, y):
        return f"{int(round(x * 10))}:{int(round(y * 10))}"
        
    center_by_key = {key_for(c["x"], c["y"]): True for c in centers.values()}
    
    outer = {}
    for tile in boundary:
        center = centers[tile["id"]]
        for vec in vectors:
            nx = center["x"] + vec["x"]
            ny = center["y"] + vec["y"]
            key = key_for(nx, ny)
            if key in center_by_key:
                continue
            if key not in outer:
                outer[key] = {"x": nx, "y": ny}
                
    points = list(outer.values())
    # Sort clockwise
    points.sort(key=lambda p: math.atan2(p["y"] - board_center["y"], p["x"] - board_center["x"]))
    
    if not points:
        return []
        
    # Find start slot (top-left-most)
    best_idx = 0
    best_slot = points[0]
    for index in range(1, len(points)):
        p = points[index]
        dy = p["y"] - best_slot["y"]
        if abs(dy) > 40.0:
            if dy < 0:
                best_idx = index
                best_slot = p
        else:
            dx = p["x"] - best_slot["x"]
            if dx < 0:
                best_idx = index
                best_slot = p
                
    points = points[best_idx:] + points[:best_idx]
    
    if len(points) != slot_count:
        sampled = []
        for i in range(slot_count):
            idx = math.floor(i * len(points) / float(slot_count))
            sampled.append(points[idx])
        points = sampled
        
    return [{
        "slotIndex": i,
        "x": p["x"],
        "y": p["y"],
        "angle": math.atan2(p["y"] - board_center["y"], p["x"] - board_center["x"])
    } for i, p in enumerate(points)]

def score_center_quality(bounds, markers_found, refine_metrics):
    area_ratio = (bounds["w"] * bounds["h"]) / (float(NORMALIZED_BOARD_SIZE) * NORMALIZED_BOARD_SIZE)
    marker_score = clamp((markers_found - CENTER_QUALITY_THRESHOLDS["markerCountWarn"]) /
                         float(CENTER_QUALITY_THRESHOLDS["markerCountGood"] - CENTER_QUALITY_THRESHOLDS["markerCountWarn"]), 0.0, 1.0)
    area_score = clamp(1.0 - abs(0.56 - area_ratio) * 2.2, 0.0, 1.0)
    
    residual = refine_metrics["meanResidualPx"] if refine_metrics and "meanResidualPx" in refine_metrics else NORMALIZED_BOARD_SIZE * 0.2
    inlier_ratio = refine_metrics["inlierRatio"] if refine_metrics and "inlierRatio" in refine_metrics else 0.0
    
    residual_score = clamp(1.0 - residual / 16.0, 0.0, 1.0)
    inlier_score = clamp((inlier_ratio - 0.22) / 0.5, 0.0, 1.0)
    
    overall = clamp(marker_score * 0.2 + area_score * 0.15 + inlier_score * 0.35 + residual_score * 0.3, 0.0, 1.0)
    
    return {
        "areaRatio": area_ratio,
        "markersFound": markers_found,
        "markerScore": marker_score,
        "areaScore": area_score,
        "inlierRatio": inlier_ratio,
        "meanResidualPx": residual,
        "refineApplied": bool(refine_metrics and refine_metrics.get("applied")),
        "overall": overall
    }

def detect_token_markers(img_rgb):
    # HSV conversion
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Range limit: Saturation < 85, Value > 160
    lower = np.array([0, 0, 160], dtype=np.uint8)
    upper = np.array([180, 85, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w, _ = img_rgb.shape
    min_area = (w * h) * 0.00015
    max_area = (w * h) * 0.01
    
    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(max(1, bh))
        if aspect < 0.6 or aspect > 1.6:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > 0.55:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                markers.append({
                    "x": cx,
                    "y": cy,
                    "radius": max(bw, bh) / 2.0,
                    "area": area,
                    "fill": 1.0,
                    "roundness": circularity
                })
                
    markers.sort(key=lambda m: m["roundness"], reverse=True)
    return {
        "markers": markers[:36],
        "markerCount": min(len(markers), 36)
    }

def estimate_similarity_from_pairs(src_a, src_b, dst_a, dst_b):
    vx = src_b["x"] - src_a["x"]
    vy = src_b["y"] - src_a["y"]
    wx = dst_b["x"] - dst_a["x"]
    wy = dst_b["y"] - dst_a["y"]
    
    src_len = math.hypot(vx, vy)
    dst_len = math.hypot(wx, wy)
    if src_len < 1e-5 or dst_len < 1e-5:
        return None
        
    scale = dst_len / src_len
    if scale < 0.72 or scale > 1.4:
        return None
        
    ang_src = math.atan2(vy, vx)
    ang_dst = math.atan2(wy, wx)
    theta = ang_dst - ang_src
    cos = math.cos(theta)
    sin = math.sin(theta)
    
    tx = dst_a["x"] - scale * (cos * src_a["x"] - sin * src_a["y"])
    ty = dst_a["y"] - scale * (sin * src_a["x"] + cos * src_a["y"])
    
    return {"scale": scale, "cos": cos, "sin": sin, "tx": tx, "ty": ty}

def apply_similarity(point, t):
    px = point["x"]
    py = point["y"]
    return {
        "x": t["scale"] * (t["cos"] * px - t["sin"] * py) + t["tx"],
        "y": t["scale"] * (t["sin"] * px + t["cos"] * py) + t["ty"]
    }

def match_predicted_to_markers(predicted, markers, max_distance):
    candidates = []
    for i, p in enumerate(predicted):
        best_idx = -1
        best_dist = float('inf')
        for j, m in enumerate(markers):
            d = math.hypot(m["x"] - p["x"], m["y"] - p["y"])
            if d < best_dist:
                best_dist = d
                best_idx = j
        if best_idx >= 0 and best_dist <= max_distance:
            candidates.append({
                "predictedIndex": i,
                "markerIndex": best_idx,
                "distance": best_dist
            })
            
    candidates.sort(key=lambda c: c["distance"])
    used_markers = set()
    used_predicted = set()
    pairs = []
    for c in candidates:
        if c["markerIndex"] in used_markers or c["predictedIndex"] in used_predicted:
            continue
        used_markers.add(c["markerIndex"])
        used_predicted.add(c["predictedIndex"])
        pairs.append(c)
    return pairs

def refine_centers_by_markers(layout, frame_slots, markers):
    predicted = [{"tileId": tile_id, "x": c["x"], "y": c["y"]} for tile_id, c in layout["centers"].items()]
    
    if not markers or len(markers) < 3:
        return {
            "centers": layout["centers"],
            "frameSlots": frame_slots,
            "metrics": {
                "applied": False,
                "inlierRatio": 0.0,
                "meanResidualPx": NORMALIZED_BOARD_SIZE * 0.2,
                "inliers": [],
                "markerCount": len(markers) if markers else 0
            }
        }
        
    initial_pairs = match_predicted_to_markers(predicted, markers, NORMALIZED_BOARD_SIZE * 0.2)
    if len(initial_pairs) < 3:
        return {
            "centers": layout["centers"],
            "frameSlots": frame_slots,
            "metrics": {
                "applied": False,
                "inlierRatio": 0.0,
                "meanResidualPx": NORMALIZED_BOARD_SIZE * 0.2,
                "inliers": [],
                "markerCount": len(markers)
            }
        }
        
    max_iters = min(220, len(initial_pairs) * len(initial_pairs))
    inlier_threshold = NORMALIZED_BOARD_SIZE * 0.065
    best_score: float = -1.0
    best_t: Optional[Dict[str, float]] = None
    best_inliers: List[Any] = []
    best_inlier_count: int = 0
    best_mean_residual: float = 0.0
    
    import random
    rng = random.Random(42) # Deterministic random
    
    for _ in range(max_iters):
        a = rng.choice(initial_pairs)
        b = rng.choice(initial_pairs)
        if a == b:
            continue
        src_a = predicted[a["predictedIndex"]]
        src_b = predicted[b["predictedIndex"]]
        dst_a = markers[a["markerIndex"]]
        dst_b = markers[b["markerIndex"]]
        t = estimate_similarity_from_pairs(src_a, src_b, dst_a, dst_b)
        if not t:
            continue
            
        inlier_count = 0
        residual_sum = 0.0
        inliers = []
        for pair in initial_pairs:
            src = predicted[pair["predictedIndex"]]
            pred = apply_similarity(src, t)
            dst = markers[pair["markerIndex"]]
            d = math.hypot(pred["x"] - dst["x"], pred["y"] - dst["y"])
            if d <= inlier_threshold:
                inlier_count += 1
                residual_sum += d
                inliers.append(pair)
                
        if inlier_count < 3:
            continue
        mean_residual = residual_sum / inlier_count
        score = inlier_count * 1000 - mean_residual
        if best_t is None or score > best_score:
            best_score = score
            best_t = t
            best_inliers = inliers
            best_inlier_count = inlier_count
            best_mean_residual = mean_residual
            
    if best_t is None or best_inlier_count < 3:
        return {
            "centers": layout["centers"],
            "frameSlots": frame_slots,
            "metrics": {
                "applied": False,
                "inlierRatio": 0.0,
                "meanResidualPx": NORMALIZED_BOARD_SIZE * 0.2,
                "inliers": [],
                "markerCount": len(markers)
            }
        }
        
    refined_centers = {}
    for tile_id, c in layout["centers"].items():
        refined_centers[tile_id] = apply_similarity(c, best_t)
        
    refined_slots = []
    for slot in frame_slots:
        p = apply_similarity(slot, best_t)
        slot_copy = dict(slot)
        slot_copy["x"] = p["x"]
        slot_copy["y"] = p["y"]
        refined_slots.append(slot_copy)
        
    return {
        "centers": refined_centers,
        "frameSlots": refined_slots,
        "metrics": {
            "applied": True,
            "inlierRatio": best_inlier_count / float(max(1, len(initial_pairs))),
            "meanResidualPx": best_mean_residual,
            "inliers": best_inliers,
            "markerCount": len(markers)
        }
    }

def hue_delta(a, b):
    d = abs(a - b)
    if d > 180.0:
        d = 360.0 - d
    return d

def detect_hex_patch_centroids(img_rgb, geometry):
    target_w = 360
    h_orig, w_orig, _ = img_rgb.shape
    scale_down = max(1.0, w_orig / float(target_w))
    w = max(1, int(round(w_orig / scale_down)))
    h = max(1, int(round(h_orig / scale_down)))
    
    sample_img = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
    
    # Build HSV buffer
    hsv_buf = np.zeros(w * h * 3, dtype=np.float32)
    for py in range(h):
        for px in range(w):
            color = sample_img[py, px]
            hue_val, s_val, v_val = rgb_to_hsv(color[0], color[1], color[2])
            idx = (py * w + px) * 3
            hsv_buf[idx] = hue_val
            hsv_buf[idx+1] = s_val
            hsv_buf[idx+2] = v_val
            
    visited = np.zeros(w * h, dtype=np.uint8)
    centroids = []
    
    scaled_hex_area = (geometry["hexW"] * geometry["hexH"] * 0.74) / (scale_down * scale_down)
    min_area = max(90, int(round(scaled_hex_area * 0.24)))
    max_area = max(min_area + 1, int(round(scaled_hex_area * 1.55)))
    
    def is_ocean_hsv(hue, sat, val):
        return hue >= 184 and hue <= 228 and sat >= 0.22 and val >= 0.12
        
    for y in range(h):
        for x in range(w):
            start = y * w + x
            if visited[start]:
                continue
            sidx = start * 3
            sh, ss, sv = hsv_buf[sidx], hsv_buf[sidx+1], hsv_buf[sidx+2]
            if ss < 0.11 or sv < 0.1 or is_ocean_hsv(sh, ss, sv):
                visited[start] = 1
                continue
                
            visited[start] = 1
            queue = [start]
            seed = {"hue": sh, "sat": ss, "val": sv}
            
            count = 0
            sum_x = 0
            sum_y = 0
            min_x = x
            max_x = x
            min_y = y
            max_y = y
            
            head = 0
            while head < len(queue):
                idx = queue[head]
                head += 1
                
                px = idx % w
                py = idx // w
                count += 1
                sum_x += px
                sum_y += py
                min_x = min(min_x, px)
                max_x = max(max_x, px)
                min_y = min(min_y, py)
                max_y = max(max_y, py)
                
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        nidx = ny * w + nx
                        if not visited[nidx]:
                            base = nidx * 3
                            hue = hsv_buf[base]
                            sat = hsv_buf[base + 1]
                            val = hsv_buf[base + 2]
                            if sat >= 0.11 and val >= 0.1 and not is_ocean_hsv(hue, sat, val):
                                if hue_delta(hue, seed["hue"]) <= 17.0 and abs(sat - seed["sat"]) <= 0.2 and abs(val - seed["val"]) <= 0.24:
                                    visited[nidx] = 1
                                    queue.append(nidx)
                                    
            if count < min_area or count > max_area:
                continue
                
            bw = max_x - min_x + 1
            bh = max_y - min_y + 1
            fill = count / float(max(1, bw * bh))
            aspect = bw / float(max(1, bh))
            if fill < 0.18 or aspect < 0.56 or aspect > 1.8:
                continue
                
            centroids.append({
                "x": (sum_x / float(max(1, count))) * scale_down,
                "y": (sum_y / float(max(1, count))) * scale_down,
                "area": count
            })
            
    centroids.sort(key=lambda c: c["area"], reverse=True)
    return centroids[:64]

def sample_hex_patch_score(img_hsv_buf, w_buf, h_buf, x, y, geometry):
    outer_r = geometry["hexW"] * 0.34
    inner_r = geometry["hexW"] * 0.2
    count = 24
    valid = 0
    hue_x = 0.0
    hue_y = 0.0
    sat_sum = 0.0
    val_sum = 0.0
    ocean_hits = 0
    
    for i in range(count):
        a = (math.pi * 2 * i) / count
        r = inner_r if i % 2 == 0 else outer_r
        sx = int(round(x + math.cos(a) * r))
        sy = int(round(y + math.sin(a) * r))
        if sx < 0 or sy < 0 or sx >= w_buf or sy >= h_buf:
            continue
            
        idx = (sy * w_buf + sx) * 3
        h = img_hsv_buf[idx]
        s = img_hsv_buf[idx + 1]
        v = img_hsv_buf[idx + 2]
        
        if s < 0.09 or v < 0.08:
            continue
            
        if h >= 182 and h <= 228 and s >= 0.22 and v >= 0.1:
            ocean_hits += 1
            continue
            
        rad = (h * math.pi) / 180.0
        hue_x += math.cos(rad)
        hue_y += math.sin(rad)
        sat_sum += s
        val_sum += v
        valid += 1
        
    if valid < 9:
        return -1.0
        
    coherence = math.hypot(hue_x, hue_y) / valid
    sat_avg = sat_sum / valid
    val_avg = val_sum / valid
    ocean_penalty = ocean_hits / float(count)
    return coherence * 0.6 + sat_avg * 0.25 + val_avg * 0.15 - ocean_penalty * 0.35

def fit_similarity_from_matches(predicted, markers, pairs, max_iters, inlier_threshold) -> Optional[FitResult]:
    if not pairs or len(pairs) < 3:
        return None
    best_score: float = -1.0
    best_t: Optional[Dict[str, float]] = None
    best_inlier_count: int = 0
    best_mean_residual: float = 0.0
    
    import random
    rng = random.Random(42)
    
    for _ in range(max_iters):
        a = rng.choice(pairs)
        b = rng.choice(pairs)
        if a == b:
            continue
        src_a = predicted[a["predictedIndex"]]
        src_b = predicted[b["predictedIndex"]]
        dst_a = markers[a["markerIndex"]]
        dst_b = markers[b["markerIndex"]]
        t = estimate_similarity_from_pairs(src_a, src_b, dst_a, dst_b)
        if not t:
            continue
            
        inlier_count = 0
        residual_sum = 0.0
        for pair in pairs:
            src = predicted[pair["predictedIndex"]]
            pred = apply_similarity(src, t)
            dst = markers[pair["markerIndex"]]
            d = math.hypot(pred["x"] - dst["x"], pred["y"] - dst["y"])
            if d <= inlier_threshold:
                inlier_count += 1
                residual_sum += d
                
        if inlier_count < 3:
            continue
        mean_residual = residual_sum / inlier_count
        score = inlier_count * 1000 - mean_residual
        if best_t is None or score > best_score:
            best_score = score
            best_t = t
            best_inlier_count = inlier_count
            best_mean_residual = mean_residual
            
    if best_t is None:
        return None

    return {
        "t": best_t,
        "inlierCount": best_inlier_count,
        "meanResidual": best_mean_residual,
        "score": best_score
    }

def refine_centers_by_color_patches(img_rgb, centers, geometry):
    h, w, _ = img_rgb.shape
    
    # Build HSV buffer using vectorized OpenCV C++ cvtColor
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] *= 2.0  # Map 0..180 to 0..360
    hsv[:, :, 1] /= 255.0 # Map 0..255 to 0..1
    hsv[:, :, 2] /= 255.0 # Map 0..255 to 0..1
    hsv_buf = hsv.flatten()
            
    patch_centroids = detect_hex_patch_centroids(img_rgb, geometry)
    predicted = [{"tileId": tile_id, "x": c["x"], "y": c["y"]} for tile_id, c in centers.items()]
    patch_pairs = match_predicted_to_markers(predicted, patch_centroids, geometry["hStep"] * 0.34)
    
    max_iters = min(260, max(40, len(patch_pairs) * len(patch_pairs)))
    fit: Optional[FitResult] = fit_similarity_from_matches(
        predicted,
        patch_centroids,
        patch_pairs,
        max_iters,
        geometry["hStep"] * 0.22
    )
    
    globally_aligned = dict(centers)
    global_applied = False
    if fit is not None and fit["inlierCount"] >= 3 and 0.82 <= fit["t"]["scale"] <= 1.28:
        globally_aligned = {}
        for tile_id, c in centers.items():
            globally_aligned[tile_id] = apply_similarity(c, fit["t"])
        global_applied = True
        
    # Outward bias for top and bottom rows
    outward_anchored = dict(globally_aligned)
    try:
        tile_rows = {}
        min_row = float('inf')
        max_row = float('-inf')
        mode_rows = geometry.get("rows") or [3, 4, 5, 4, 3]
        
        idx = 0
        for r_idx, row_count in enumerate(mode_rows):
            for col in range(row_count):
                tile_rows[idx] = r_idx
                min_row = min(min_row, r_idx)
                max_row = max(max_row, r_idx)
                idx += 1
                
        sum_x = sum(c["x"] for c in globally_aligned.values())
        sum_y = sum(c["y"] for c in globally_aligned.values())
        n = len(globally_aligned)
        board_center = {"x": sum_x / n, "y": sum_y / n}
        
        outward_bias = 0.13
        for pair in patch_pairs:
            tile_id = predicted[pair["predictedIndex"]]["tileId"]
            row = tile_rows.get(tile_id)
            if row == min_row or row == max_row:
                center = globally_aligned[tile_id]
                patch = patch_centroids[pair["markerIndex"]]
                vx = patch["x"] - board_center["x"]
                vy = patch["y"] - board_center["y"]
                outward_anchored[tile_id] = {
                    "x": center["x"] + vx * outward_bias,
                    "y": center["y"] + vy * outward_bias
                }
    except Exception as e:
        pass
        
    anchored = dict(outward_anchored)
    for pair in patch_pairs:
        tile = predicted[pair["predictedIndex"]]
        patch = patch_centroids[pair["markerIndex"]]
        old = anchored[tile["tileId"]]
        anchored[tile["tileId"]] = {
            "x": old["x"] * 0.55 + patch["x"] * 0.45,
            "y": old["y"] * 0.55 + patch["y"] * 0.45
        }
        
    search_r = max(8, int(round(min(geometry["hStep"], geometry["vStep"]) * 0.34)))
    step = 3
    refined = {}
    moved = 0
    shift_sum = 0.0
    
    for tile_id, center in anchored.items():
        best_x = center["x"]
        best_y = center["y"]
        best_score = sample_hex_patch_score(hsv_buf, w, h, center["x"], center["y"], geometry)
        
        for dy in range(-search_r, search_r + 1, step):
            for dx in range(-search_r, search_r + 1, step):
                px = center["x"] + dx
                py = center["y"] + dy
                if px < 0 or py < 0 or px >= w or py >= h:
                    continue
                dist_penalty = math.hypot(dx, dy) / float(max(1, search_r))
                score = sample_hex_patch_score(hsv_buf, w, h, px, py, geometry) - dist_penalty * 0.18
                if score > best_score:
                    best_score = score
                    best_x = px
                    best_y = py
                    
        shift = math.hypot(best_x - center["x"], best_y - center["y"])
        if shift > 0.75:
            moved += 1
            shift_sum += shift
        refined[tile_id] = {"x": best_x, "y": best_y}
        
    return {
        "centers": refined,
        "metrics": {
            "moved": moved,
            "matchedPatchCentroids": len(patch_pairs),
            "globalPatchFitApplied": global_applied,
            "globalPatchFitScale": fit["t"]["scale"] if fit is not None else 1.0,
            "globalPatchFitResidual": fit["meanResidual"] if fit is not None else 0.0,
            "avgShiftPx": shift_sum / moved if moved > 0 else 0.0,
            "applied": moved > 0
        }
    }



def fit_weighted_axis_aligned_transform(src_points, dst_points, weights=None):
    """
    Computes optimal 2D Scale and Translation (WITHOUT rotation/tilt)
    mapping src_points to dst_points via Weighted Least Squares.
    Forces hex grid to remain strictly upright (zero tilt).
    """
    n = len(src_points)
    if n < 1:
        return None

    if weights is None:
        weights = [1.0] * n

    w_sum = sum(weights)
    if w_sum <= 0:
        return None

    p_bar_x = sum(w * p["x"] for w, p in zip(weights, src_points)) / w_sum
    p_bar_y = sum(w * p["y"] for w, p in zip(weights, src_points)) / w_sum

    q_bar_x = sum(w * q["x"] for w, q in zip(weights, dst_points)) / w_sum
    q_bar_y = sum(w * q["y"] for w, q in zip(weights, dst_points)) / w_sum

    denom = sum(w * ((p["x"] - p_bar_x)**2 + (p["y"] - p_bar_y)**2) for w, p in zip(weights, src_points))
    num = sum(w * ((p["x"] - p_bar_x) * (q["x"] - q_bar_x) + (p["y"] - p_bar_y) * (q["y"] - q_bar_y)) for w, p, q in zip(weights, src_points, dst_points))

    scale = (num / denom) if denom > 1e-6 else 1.0
    scale = max(0.6, min(1.8, scale))

    tx = q_bar_x - scale * p_bar_x
    ty = q_bar_y - scale * p_bar_y

    return {
        "scale": scale,
        "cos": 1.0,
        "sin": 0.0,
        "tx": tx,
        "ty": ty
    }

def estimate_axis_aligned_from_pairs(src_a, src_b, dst_a, dst_b):
    vx = src_b["x"] - src_a["x"]
    vy = src_b["y"] - src_a["y"]
    wx = dst_b["x"] - dst_a["x"]
    wy = dst_b["y"] - dst_a["y"]

    src_len = math.hypot(vx, vy)
    dst_len = math.hypot(wx, wy)
    if src_len < 1e-5 or dst_len < 1e-5:
        return None

    scale = dst_len / src_len
    if scale < 0.72 or scale > 1.4:
        return None

    tx = dst_a["x"] - scale * src_a["x"]
    ty = dst_a["y"] - scale * src_a["y"]

    return {"scale": scale, "cos": 1.0, "sin": 0.0, "tx": tx, "ty": ty}

def normalize_centers_to_ideal_grid(layout, frame_slots, candidate_centers, markers=None):
    """
    Fits the canonical ideal hex layout onto detected candidate centers using
    RANSAC + Weighted Least-Squares Axis-Aligned Transformation (zero tilt).
    Enforces that all 19 tile centers are perfectly equidistant and upright.
    """
    canonical = layout["centers"]
    tile_ids = sorted(canonical.keys())

    if not candidate_centers:
        return {"centers": canonical, "frameSlots": frame_slots}

    src_list = [canonical[tid] for tid in tile_ids]
    dst_list = [candidate_centers.get(tid, canonical[tid]) for tid in tile_ids]

    weights = []
    marker_list = markers if markers else []
    for tid in tile_ids:
        dst = candidate_centers.get(tid, canonical[tid])
        w = 1.0
        for m in marker_list:
            if math.hypot(m["x"] - dst["x"], m["y"] - dst["y"]) <= 40.0:
                w = 2.5
                break
        weights.append(w)

    import random
    rng = random.Random(42)

    inlier_threshold = NORMALIZED_BOARD_SIZE * 0.05
    best_inliers = []
    best_t = None

    for _ in range(120):
        idx1, idx2 = rng.sample(range(len(tile_ids)), 2)
        p1, p2 = src_list[idx1], src_list[idx2]
        q1, q2 = dst_list[idx1], dst_list[idx2]
        t = estimate_axis_aligned_from_pairs(p1, p2, q1, q2)
        if not t:
            continue

        inliers = []
        for k in range(len(tile_ids)):
            pred = apply_similarity(src_list[k], t)
            d = math.hypot(pred["x"] - dst_list[k]["x"], pred["y"] - dst_list[k]["y"])
            if d <= inlier_threshold:
                inliers.append(k)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_t = t

    if len(best_inliers) >= 3:
        inlier_src = [src_list[k] for k in best_inliers]
        inlier_dst = [dst_list[k] for k in best_inliers]
        inlier_w = [weights[k] for k in best_inliers]

        refined_t = fit_weighted_axis_aligned_transform(inlier_src, inlier_dst, inlier_w)
        if refined_t:
            best_t = refined_t

    if not best_t:
        best_t = fit_weighted_axis_aligned_transform(src_list, dst_list, weights)

    if not best_t:
        return {"centers": candidate_centers, "frameSlots": frame_slots}

    normalized_centers = {}
    for tid, c in canonical.items():
        normalized_centers[tid] = apply_similarity(c, best_t)

    normalized_slots = []
    for slot in frame_slots:
        p = apply_similarity(slot, best_t)
        s_copy = dict(slot)
        s_copy["x"] = p["x"]
        s_copy["y"] = p["y"]
        normalized_slots.append(s_copy)

    return {
        "centers": normalized_centers,
        "frameSlots": normalized_slots,
        "transform": best_t
    }

def detect_centers(img_rgb, mode_key):
    mode = "six" if mode_key == "six" else "four"
    h, w, _ = img_rgb.shape
    
    bounds = detect_board_bounds(img_rgb)
    size = bounds["w"]
    
    # Safe bounds checking and square cropping
    x0 = max(0, bounds["x"])
    y0 = max(0, bounds["y"])
    x1 = min(w, bounds["x"] + size)
    y1 = min(h, bounds["y"] + size)
    
    cropped = img_rgb[y0:y1, x0:x1]
    if cropped.size == 0:
        cropped = img_rgb
        bounds = {"x": 0, "y": 0, "w": w, "h": h}
        size = max(w, h)
        
    crop_h, crop_w, _ = cropped.shape
    if crop_h != size or crop_w != size:
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        dy = max(0, -bounds["y"])
        dx = max(0, -bounds["x"])
        padded[dy:min(size, dy+crop_h), dx:min(size, dx+crop_w)] = cropped[:size-dy, :size-dx]
        cropped = padded
        
    normalized_img = cv2.resize(cropped, (NORMALIZED_BOARD_SIZE, NORMALIZED_BOARD_SIZE), interpolation=cv2.INTER_LINEAR)

    
    normalized_bounds = {"x": 0, "y": 0, "w": NORMALIZED_BOARD_SIZE, "h": NORMALIZED_BOARD_SIZE}
    layout = build_layout(MODE_ROWS[mode], normalized_bounds)
    frame_slots = build_frame_slots(
        layout["tiles"],
        layout["adjacency"],
        layout["centers"],
        layout["boardCenter"],
        layout["geometry"],
        MODE_FRAME_SLOTS[mode]
    )
    
    marker_info = detect_token_markers(normalized_img)
    refined = refine_centers_by_markers(layout, frame_slots, marker_info["markers"])
    color_patch_refined = refine_centers_by_color_patches(normalized_img, refined["centers"], layout["geometry"])
    
    # Global regular hex grid normalization to ensure all centers are perfectly equidistant
    grid_normalized = normalize_centers_to_ideal_grid(
        layout,
        refined["frameSlots"],
        color_patch_refined["centers"],
        marker_info["markers"]
    )
    final_centers = grid_normalized["centers"]
    final_frame_slots = grid_normalized["frameSlots"]

    quality = score_center_quality(normalized_bounds, marker_info["markerCount"], refined["metrics"])
    
    spiral_order = build_spiral_order(
        layout["tiles"],
        layout["adjacency"],
        final_centers,
        layout["boardCenter"]
    )
    
    centers = []
    for tile in layout["tiles"]:
        center = final_centers[tile["id"]]
        centers.append({
            "tileId": tile["id"],
            "row": tile["row"],
            "col": tile["col"],
            "x": center["x"],
            "y": center["y"]
        })
        
    # Return everything we need
    return {
        "modeKey": mode,
        "bounds": {
            "original": bounds,
            "normalized": normalized_bounds
        },
        "centers": centers,
        "frameSlots": final_frame_slots,
        "spiralOrder": spiral_order,
        "geometry": layout["geometry"],
        "markerDebug": {
            "markers": marker_info["markers"],
            "refinement": refined["metrics"],
            "colorPatchRefinement": color_patch_refined["metrics"]
        },
        "quality": quality,
        "normalizedImage": normalized_img # Return standard numpy array of 960x960
    }
