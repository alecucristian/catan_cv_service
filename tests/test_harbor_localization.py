import os
import cv2
import pytest
import numpy as np

from app.cv.centers import detect_centers
from tests.conftest import get_test_image_paths

test_images = get_test_image_paths()


@pytest.mark.parametrize(
    "image_path",
    test_images,
    ids=[os.path.basename(p) for p in test_images],
)
def test_harbor_slot_localization(image_path: str):
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]

    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Could not read {image_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    center_result = detect_centers(img_rgb, "four")
    frame_slots = center_result["frameSlots"]
    norm_img = center_result["normalizedImage"]
    h_img, w_img, _ = norm_img.shape
    hex_w = center_result["geometry"]["hexW"]

    # In 4-player standard layout, active harbors are at even indices (0, 2, 4, 6, 8, 10, 12, 14, 16)
    even_slots = [s for s in frame_slots if s["slotIndex"] % 2 == 0]
    assert len(even_slots) == 9, f"Expected 9 active harbor slots, found {len(even_slots)}"

    gray_img = cv2.cvtColor(norm_img, cv2.COLOR_RGB2GRAY)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "test_outputs",
        "harbor_localization",
        base_name[:20],
    )
    os.makedirs(output_dir, exist_ok=True)

    for slot in even_slots:
        slot_idx = slot["slotIndex"]
        sx, sy = int(round(slot["x"])), int(round(slot["y"]))

        # Verify initial coordinates are inside image bounds
        assert 0 <= sx < w_img, f"Slot {slot_idx} X ({sx}) is outside image width ({w_img})"
        assert 0 <= sy < h_img, f"Slot {slot_idx} Y ({sy}) is outside image height ({h_img})"

        # Sail search region
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

        # Extract harbor crop
        span = int(round(hex_w * 0.45))
        y0 = max(0, snap_y - span // 2)
        y1 = min(h_img, snap_y + span // 2)
        x0 = max(0, snap_x - span // 2)
        x1 = min(w_img, snap_x + span // 2)
        cropped = norm_img[y0:y1, x0:x1]

        assert cropped.size > 0, f"Harbor crop for slot {slot_idx} is empty!"

        # Save cropped debug image
        crop_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"slot_{slot_idx:02d}.png"), crop_bgr)

        # Check maximum luminance (sail must be bright)
        max_luma = np.max(cropped)
        assert max_luma >= 150, f"Slot {slot_idx} max luminance {max_luma} is too low for harbor sail!"
