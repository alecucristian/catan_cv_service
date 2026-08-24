import pytest
import os
import glob

def pytest_addoption(parser):
    parser.addoption(
        "--save-overlays",
        action="store_true",
        default=False,
        help="Generate and save visual debug overlay images to test_outputs/",
    )

@pytest.fixture
def save_overlays(request) -> bool:
    return request.config.getoption("--save-overlays")

def get_test_image_paths() -> list[str]:
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_images")
    if not os.path.exists(test_dir):
        return []
    image_paths = []
    for ext in ("*.jpeg", "*.jpg", "*.png", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
    return sorted(image_paths)
