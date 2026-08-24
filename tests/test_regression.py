import os
import cv2
import pytest

from app.cv.pipeline import run_pipeline
from tests.conftest import get_test_image_paths
from tests.helpers import (
    parse_board_code,
    parse_harbor_map,
    draw_detection_overlay,
    HARBOR_CODE_TO_LABEL,
)

test_images = get_test_image_paths()


@pytest.mark.parametrize(
    "image_path",
    test_images,
    ids=[os.path.basename(p) for p in test_images],
)
def test_board_image_detection(image_path: str, save_overlays: bool):
    filename = os.path.basename(image_path)
    expected_code = os.path.splitext(filename)[0]

    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Failed to load image from: {image_path}"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = run_pipeline(img_rgb, "four")

    detected_code = result.get("boardCode", "")
    det_clean = detected_code.replace(" ", "")
    exp_clean = expected_code.replace(" ", "")

    exp_tiles, exp_ports = parse_board_code(expected_code)
    det_tiles, det_ports = parse_board_code(detected_code)

    exp_h_map = parse_harbor_map(exp_ports)
    det_h_map = parse_harbor_map(det_ports)

    # 1. Tile mismatches
    tile_mismatches = []
    max_tiles = max(len(exp_tiles), len(det_tiles))
    for idx in range(max_tiles):
        exp_t = exp_tiles[idx] if idx < len(exp_tiles) else "<missing>"
        det_t = det_tiles[idx] if idx < len(det_tiles) else "<missing>"
        if exp_t != det_t:
            tile_mismatches.append(f"  - Tile #{idx:02d}: Expected '{exp_t}', Detected '{det_t}'")

    # 2. Harbor mismatches
    harbor_mismatches = []
    all_slots = sorted(set(exp_h_map.keys()) | set(det_h_map.keys()))
    for slot in all_slots:
        exp_h = exp_h_map.get(slot)
        det_h = det_h_map.get(slot)
        if exp_h != det_h:
            exp_desc = f"'{exp_h}' ({HARBOR_CODE_TO_LABEL.get(exp_h, exp_h)})" if exp_h else "<missing>"
            det_desc = f"'{det_h}' ({HARBOR_CODE_TO_LABEL.get(det_h, det_h)})" if det_h else "<missing>"
            harbor_mismatches.append(f"  - Harbor Slot {slot:02d}: Expected {exp_desc}, Detected {det_desc}")

    # Optionally save visual overlay
    if save_overlays:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"output_{filename}")
        annotated_img = draw_detection_overlay(img_bgr, result, expected_code=expected_code, filename=filename)
        cv2.imwrite(output_path, annotated_img)

    # Build granular diagnostic report if mismatched
    if tile_mismatches or harbor_mismatches:
        report_lines = [
            f"Board recognition mismatch for {filename}:",
            f"  Expected: {exp_clean}",
            f"  Detected: {det_clean}",
        ]
        if tile_mismatches:
            report_lines.append(f"\nTile Discrepancies ({len(tile_mismatches)}):")
            report_lines.extend(tile_mismatches)
        if harbor_mismatches:
            report_lines.append(f"\nHarbor Discrepancies ({len(harbor_mismatches)}):")
            report_lines.extend(harbor_mismatches)

        pytest.fail("\n".join(report_lines))
