import torch

from PIL.Image import Image
from torchvision import transforms
from PIL import Image
from typing import Any, Optional
from functools import lru_cache


class ResnetSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._init_model()
        return cls._instance

    @staticmethod
    def _init_model():
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
        return model


def get_rectangle_coordinate(det: list[int]) -> list[int]:
    return [det[0], det[1], det[0] + det[2],
            det[1] + det[3]]


def compute_iou(boxA: list[int], boxB: list[int]) -> float:
    boxA, boxB = get_rectangle_coordinate(boxA), get_rectangle_coordinate(boxB)
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def preprocess_image(image_path: str, x: int, y: int, width: int, height: int) -> Image:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = image.crop((x, y, x + width, y + height))  # Crop based on coordinates
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image


@lru_cache(maxsize=None)
def extract_features(image_path: str, x: int, y: int, width: int, height: int) -> torch.Tensor:
    model = ResnetSingleton.get_instance()
    image = preprocess_image(image_path, x, y, width, height)
    with torch.no_grad():
        features = model(image)
    return features.squeeze()
