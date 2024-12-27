from dataclasses import dataclass
from pathlib import Path

import yaml
import torch
from torchvision import transforms
from .manual_models import ResNetWithEmbeddings, EfficientNetWithEmbeddings
from PIL import Image

# load config file
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


@dataclass
class ROPPrediction:
    """Class representing a ROP prediction result."""

    label: int
    confidence: float


def load_model():
    """Load a pre-trained ROP classification model.

    Returns:
        model (function): A function that takes a photo of the fundus and returns a disease prediction object.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["model"] == "efficientnet":
        model = EfficientNetWithEmbeddings(2)
        state = torch.load(Path(__file__).parent / 'state_models/efficientnet.pth', map_location=device)
    elif config["model"] == "resnet18":
        model = ResNetWithEmbeddings(2)
        state = torch.load(Path(__file__).parent / 'state_models/resnet18.pth', map_location=device)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    model.load_state_dict(state)
    model.eval()

    transform_img = transforms.Compose(
        [
            transforms.Resize(size=(480, 640)),
            transforms.ToTensor(),
        ]
    )

    def predict(img: Image.Image) -> ROPPrediction:
        """ Calculate probability of having eye disease """
        img = transform_img(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            pred = outputs.softmax(dim=1)
        return ROPPrediction(
            label=predicted.item(),
            confidence=pred[0, predicted].item(),
        )

    return predict
