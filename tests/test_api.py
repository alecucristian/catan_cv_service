import io
import os
import pytest
from fastapi import UploadFile

from app.main import app, health_check, detect_board
from tests.conftest import get_test_image_paths

test_images = get_test_image_paths()

def test_api_health():
    res = health_check()
    assert res == {"status": "ok"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "image_path",
    test_images,
    ids=[os.path.basename(p) for p in test_images],
)
async def test_api_detect_board(image_path: str):
    filename = os.path.basename(image_path)
    expected_code = os.path.splitext(filename)[0]

    with open(image_path, "rb") as f:
        file_bytes = f.read()

    upload_file = UploadFile(
        file=io.BytesIO(file_bytes),
        filename=filename,
    )

    result = await detect_board(image=upload_file, mode="four")

    detected_code = result.get("boardCode", "")
    det_clean = detected_code.replace(" ", "")
    exp_clean = expected_code.replace(" ", "")

    assert det_clean == exp_clean, f"API mismatch for {filename}:\n  Expected: {exp_clean}\n  Detected: {det_clean}"

