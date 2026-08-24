import re
import cv2
import numpy as np

# Mapping from single-letter code to full resource name
RESOURCE_CODE_TO_NAME = {
    "S": "sheep",
    "O": "ore",
    "G": "wheat",
    "W": "wood",
    "B": "brick",
    "D": "desert",
}

NAME_TO_RESOURCE_CODE = {
    "sheep": "S",
    "ore": "O",
    "wheat": "G",
    "wood": "W",
    "brick": "B",
    "desert": "D",
}

HARBOR_CODE_TO_LABEL = {
    "T": "3:1",
    "W": "wood 2:1",
    "B": "brick 2:1",
    "S": "sheep 2:1",
    "G": "wheat 2:1",
    "O": "ore 2:1",
}

LABEL_TO_HARBOR_CODE = {
    "3:1": "T",
    "wood 2:1": "W",
    "brick 2:1": "B",
    "sheep 2:1": "S",
    "wheat 2:1": "G",
    "ore 2:1": "O",
}


def parse_board_code(code_str: str):
    """
    Parses a board code string into tile codes and harbor mapping.
    Format: <Tiles> P0 <Letter><Slot>...<Slot0Letter>
    E.g. S9O10S8...P0W2T4G6T8T10O12B14T16S
    """
    clean = code_str.replace(" ", "")
    if "P" in clean:
        tile_part, harbor_part = clean.split("P", 1)
    else:
        tile_part, harbor_part = clean, ""

    tiles = [f"{r}{num}" for r, num in re.findall(r'([SOGWBD])(\d*)', tile_part)]
    
    # Strip leading '0' prefix if present from P0
    if harbor_part.startswith("0"):
        harbor_body = harbor_part[1:]
    else:
        harbor_body = harbor_part

    # Find pairs of (Letter, SlotNumber)
    pairs = re.findall(r'([TWBSGO])(\d+)', harbor_body)
    harbors = [f"{int(slot)}{htype}" for htype, slot in pairs]

    # Check for trailing slot 0 letter
    m0 = re.search(r'([TWBSGO])$', harbor_body)
    if m0:
        harbors.append(f"0{m0.group(1)}")

    return tiles, harbors


def parse_harbor_map(port_list: list[str]) -> dict[int, str]:
    """
    Parses harbor code list into a map of slot_index -> harbor_type_code (e.g. 2 -> 'W', 0 -> 'S').
    """
    h_map = {}
    for p in port_list:
        m = re.match(r'(\d+)([TWBSGO])', p)
        if m:
            h_map[int(m.group(1))] = m.group(2)
    return h_map



def fit_text_scale(text: str, font: int, initial_scale: float, max_width: int, min_scale: float = 0.2, thickness: int = 1):
    """
    Calculates font_scale and text size ensuring the text width does not exceed max_width.
    """
    scale = initial_scale
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw > max_width and tw > 0:
        scale = max(min_scale, scale * (max_width / float(tw)))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    return scale, tw, th


def draw_detection_overlay(img_bgr: np.ndarray, result: dict, expected_code: str | None = None, filename: str = "") -> np.ndarray:
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

            # Sub-label for debug inspection
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
        label = port["label"]
        slot = slots_by_index.get(slot_index)

        norm_x = port.get("x", slot["x"] if slot else 0)
        norm_y = port.get("y", slot["y"] if slot else 0)
        px, py = to_orig(norm_x, norm_y)

        p_radius = int(round(22 * scale_x))
        p_radius = max(18, min(40, p_radius))

        cv2.circle(overlay, (px, py), p_radius, (255, 220, 0), -1)
        cv2.circle(overlay, (px, py), p_radius, (0, 0, 0), 2)

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

    # 4. Draw Header Banner
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

    title_str = f"File: {filename}" if filename else "Catan CV Detection Result"
    t_scale, tw, th = fit_text_scale(title_str, font, 0.55, max_banner_w - 180, 0.3, 2)
    cv2.putText(overlay, title_str, (15, 26), font, t_scale, (255, 255, 255), 2, cv2.LINE_AA)

    if is_match is not None:
        status_str = "MATCH: PASSED" if is_match else "MATCH: FAILED"
        status_color = (0, 255, 0) if is_match else (50, 50, 255)
        s_scale, sw, sh = fit_text_scale(status_str, font, 0.55, 160, 0.3, 2)
        cv2.putText(overlay, status_str, (w - sw - 15, 26), font, s_scale, status_color, 2, cv2.LINE_AA)

    det_str = f"Detected: {det_code}"
    d_scale, dw, dh = fit_text_scale(det_str, font, 0.45, max_banner_w, 0.2, 1)
    cv2.putText(overlay, det_str, (15, 54), font, d_scale, (0, 255, 255), 1, cv2.LINE_AA)

    if expected_code:
        exp_str = f"Expected: {expected_code}"
        e_scale, ew, eh = fit_text_scale(exp_str, font, 0.45, max_banner_w, 0.2, 1)
        cv2.putText(overlay, exp_str, (15, 80), font, e_scale, (200, 200, 200), 1, cv2.LINE_AA)

        stats_str = f"Mode: {result['modeKey']} | Hexes: {len(result['centers'])} | Ports: {len(result['ports'])} | Quality: {result['quality']['overall']:.3f}"
        st_scale, stw, sth = fit_text_scale(stats_str, font, 0.42, max_banner_w, 0.2, 1)
        cv2.putText(overlay, stats_str, (15, 104), font, st_scale, (170, 170, 170), 1, cv2.LINE_AA)
    else:
        stats_str = f"Mode: {result['modeKey']} | Hexes: {len(result['centers'])} | Ports: {len(result['ports'])} | Quality: {result['quality']['overall']:.3f}"
        st_scale, stw, sth = fit_text_scale(stats_str, font, 0.42, max_banner_w, 0.2, 1)
        cv2.putText(overlay, stats_str, (15, 80), font, st_scale, (170, 170, 170), 1, cv2.LINE_AA)

    return overlay
