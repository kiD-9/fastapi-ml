from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI
from pydantic import BaseModel

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


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    text: str
    sentiment_label: str
    sentiment_score: float


# create a route
@app.get("/")
def index():
    return {"text": "Sentiment Analysis"}


# Your FastAPI route handlers go here
@app.post("/predict")
def predict_sentiment(request: SentimentRequest):
    sentiment = model(request.text)

    response = SentimentResponse(
        text=request.text,
        sentiment_label=sentiment.label,
        sentiment_score=sentiment.score,
    )

    return response
