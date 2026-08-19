from .centers import detect_centers
from .tiles import detect_tiles
from .tokens import detect_tokens
from .harbors import detect_harbors

RESOURCE_LETTER_MAP = {
    "wood": "W",
    "brick": "B",
    "sheep": "S",
    "wheat": "G",
    "ore": "O",
    "desert": "D"
}

HARBOR_LETTER_MAP = {
    "3:1": "T",
    "wood 2:1": "W",
    "brick 2:1": "B",
    "sheep 2:1": "S",
    "wheat 2:1": "G",
    "ore 2:1": "O"
}

def generate_board_code(tiles, spiral_order, ports):
    """
    Serializes the board configuration into the Catan board code string format.
    Matches the boardCodeFromTiles function in app.js.
    """
    # Map tiles by ID
    tiles_by_id = {t["tileId"]: t for t in tiles}
    
    # Order tiles by spiral order
    ordered_tiles = []
    for tile_id in spiral_order:
        if tile_id in tiles_by_id:
            ordered_tiles.append(tiles_by_id[tile_id])
            
    tile_parts = []
    for tile in ordered_tiles:
        res = RESOURCE_LETTER_MAP.get(tile["resource"], "")
        token = tile.get("token")
        token_str = "" if token is None else str(token)
        tile_parts.append(res + token_str)
        
    tile_code = " ".join(tile_parts)
    
    # Order ports by slotIndex
    sorted_ports = sorted(ports, key=lambda p: p["slotIndex"])
    
    # Ground truth format puts slot 0 letter at end if slot 0, or formatted as LetterSlotNumber (e.g. W2 T4 G6 T8 T10 O12 B14 T16 S)
    harbor_parts = []
    slot0_code = None
    for port in sorted_ports:
        port_type = port["type"]
        h_code = HARBOR_LETTER_MAP.get(port_type, "T")
        idx = port["slotIndex"]
        if idx == 0:
            slot0_code = h_code
        else:
            harbor_parts.append(f"{h_code}{idx}")
            
    if slot0_code is not None:
        harbor_parts.append(slot0_code)
        
    harbor_code = ""
    if harbor_parts:
        harbor_code = " P0" + "".join(harbor_parts)
        
    return tile_code + harbor_code

def run_pipeline(img_rgb, mode_key):
    """
    Runs the entire computer vision pipeline on the input image.
    1. Center/bounds detection & refinement
    2. Terrain/tile classification
    3. Token number template matching
    4. Harbor type template matching
    """
    # 1. Centers Stage
    centers_result = detect_centers(img_rgb, mode_key)
    
    # 2. Tiles Stage
    tiles_result = detect_tiles(centers_result)
    
    # 3. Tokens Stage
    tokens_result = detect_tokens(centers_result, tiles_result)
    
    # 4. Harbors Stage
    harbors_result = detect_harbors(centers_result)
    
    # Compute board code
    board_code = generate_board_code(
        tokens_result["tiles"],
        centers_result["spiralOrder"],
        harbors_result["ports"]
    )
    
    # Prepare serializable result (omit numpy images)
    return {
        "boardCode": board_code,
        "modeKey": centers_result["modeKey"],
        "bounds": {
            "original": centers_result["bounds"]["original"],
            "normalized": centers_result["bounds"]["normalized"]
        },
        "centers": centers_result["centers"],
        "frameSlots": centers_result["frameSlots"],
        "spiralOrder": centers_result["spiralOrder"],
        "quality": centers_result["quality"],
        "tiles": tokens_result["tiles"],
        "tileQuality": tiles_result["quality"],
        "ports": [
            {
                "slotIndex": p["slotIndex"],
                "label": p["type"],
                "x": p["x"],
                "y": p["y"]
            } for p in harbors_result["ports"]
        ]
    }
