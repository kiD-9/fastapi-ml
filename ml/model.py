from dataclasses import dataclass
from pathlib import Path

import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load config file
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


@dataclass
class SentimentPrediction:
    """Class representing a sentiment prediction result."""

    label: str
    score: float


def load_model():
    """Load a pre-trained sentiment analysis model.

    Returns:
        model (function): A function that takes a text input and returns a SentimentPrediction object.
    """
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model"])
    if torch.cuda.is_available():
        model.cuda()

    def predict(text: str) -> SentimentPrediction:
        """ Calculate sentiment of a text. `return_type` can be 'label', 'score' or 'proba' """
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
            proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
        return SentimentPrediction(
            label=model.config.id2label[proba.argmax()],
            score=proba.dot([-1, 0, 1]),
        )

    return predict
