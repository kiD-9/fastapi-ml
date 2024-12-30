import io

import pandas as pd
import pytest
from PIL import Image

from ml.model import ROPPrediction, load_model


@pytest.fixture(scope="module")
def dummy_image():
    # Create a dummy image
    image = Image.new("RGB", (640, 480), color="red")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

@pytest.fixture(scope="module")
def model():
    return load_model()

@pytest.fixture(scope="module")
def dataset():
    labels_path = "./tests/data/ROP.csv"
    return pd.read_csv(labels_path)

def test_load_model(model):
    assert callable(model)

def test_predict_ROP(model, dummy_image):
    image = Image.open(io.BytesIO(dummy_image))
    prediction = model(image)
    assert isinstance(prediction, ROPPrediction)
    assert prediction.label in [0, 1]
    assert 0.0 <= prediction.confidence <= 1.0

def test_predict_ROP_with_dataset(model, labels):
    for idx, row in labels.iterrows():
        image = Image.open(f'./tests/data/{row["image_name"]}')
        prediction = model(image)
        assert isinstance(prediction, ROPPrediction)
        assert prediction.label in [0, 1]
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.confidence == prediction.confidence
        assert prediction.label == row["label"]

def load_test_data():
    labels = pd.read_csv("./tests/data/ROP.csv")
    test_data = []
    for idx, row in labels.iterrows():
        test_data.append((row['image_name'], row['label']))
    return test_data

@pytest.mark.parametrize("image_name, expected_label", load_test_data())
def test_predict_ROP_with_dataset(model, image_name, expected_label):
    with open("./tests/data/" + image_name, "rb") as image_file:
        image = Image.open(io.BytesIO(image_file.read()))
    prediction = model(image)
    assert isinstance(prediction, ROPPrediction)
    assert prediction.label in [0, 1]
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.label == expected_label