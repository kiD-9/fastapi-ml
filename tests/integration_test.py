import pytest
from fastapi.testclient import TestClient
from app.app import app, predict_ROP
from ml.model import load_model
from PIL import Image
import io

@pytest.fixture(scope="module")
def dummy_image():
    # Create a dummy image
    image = Image.new("RGB", (640, 480), color="red")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def test_load_model():
    model = load_model()
    assert callable(model)

def test_predict_ROP(dummy_image):
    model = load_model()
    image = Image.open(io.BytesIO(dummy_image))
    prediction = model(image)
    assert prediction.label in [0, 1]
    assert 0.0 <= prediction.confidence <= 1.0

def test_predict_api(dummy_image):
    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("dummy.png", dummy_image, "image/png")})
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["label"] in [0, 1]
        assert 0.0 <= json_response["confidence"] <= 1.0