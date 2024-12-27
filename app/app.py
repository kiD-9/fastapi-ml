from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from PIL import Image
import io

from ml.model import load_model

model: Callable = None


# Register the function to run during startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model
    model = load_model()
    yield
    # Clean up the ML models and release the resources
    model = None


app = FastAPI(lifespan=lifespan)

class ROPResponse(BaseModel):
    name_label: str
    label: int
    confidence: float


# create a route
@app.get("/")
def index():
    return {"text": "ROP prediction"}


# Your FastAPI route handlers go here
@app.post("/predict")
async def predict_ROP(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    rop_pred = model(image)

    response = ROPResponse(
        name_label = "Больной" if bool(rop_pred.label) else "Здоровый",
        label = rop_pred.label,
        confidence = rop_pred.confidence,
    )

    return response
