import os
import cv2
import pytest
import numpy as np

from app.cv.centers import detect_centers
from app.cv.harbors import detect_harbors, HARBOR_TYPES
from tests.conftest import get_test_image_paths
from tests.helpers import parse_board_code, parse_harbor_map, HARBOR_CODE_TO_LABEL

test_images = get_test_image_paths()


def test_reference_templates_exist():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    tmpl_files = {
        "3:1": "templates/harbor_3to1.png",
        "wood 2:1": "templates/harbor_wood.png",
        "brick 2:1": "templates/harbor_brick.png",
        "sheep 2:1": "templates/harbor_sheep.png",
        "wheat 2:1": "templates/harbor_grain.png",
        "ore 2:1": "templates/harbor_ore.png",
    }
    for htype, path_rel in tmpl_files.items():
        path = os.path.join(base_dir, path_rel)
        assert os.path.exists(path), f"Template file {path} missing!"
        img = cv2.imread(path)
        assert img is not None, f"Failed to load template {path}"
        assert img.shape[0] > 10 and img.shape[1] > 10, f"Template {path} too small: {img.shape}"


@pytest.mark.parametrize(
    "image_path",
    test_images,
    ids=[os.path.basename(p) for p in test_images],
)
def test_harbor_classification_scores(image_path: str):
    filename = os.path.basename(image_path)
    expected_code = os.path.splitext(filename)[0]

    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    center_result = detect_centers(img_rgb, "four")
    harbor_result = detect_harbors(center_result)

    _, exp_ports = parse_board_code(expected_code)
    exp_h_map = parse_harbor_map(exp_ports)

    detected_ports = {p["slotIndex"]: p["type"] for p in harbor_result["ports"]}

    # Check each slot against ground truth
    mismatches = []
    for slot_idx, exp_code in exp_h_map.items():
        exp_label = HARBOR_CODE_TO_LABEL.get(exp_code, exp_code)
        det_label = detected_ports.get(slot_idx, "<missing>")
        if det_label != exp_label:
            mismatches.append(f"Slot {slot_idx:02d}: expected '{exp_label}', got '{det_label}'")

    if mismatches:
        pytest.fail(f"Harbor classification mismatches in {filename}:\n" + "\n".join(mismatches))
