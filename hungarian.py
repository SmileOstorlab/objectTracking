import numpy as np
import pandas as pd
import torch
import torchvision.models as models

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from PIL import Image

from Iou import compute_iou
from Track import Track, Frame


def model_embedding(resnet: bool):
    if resnet:
        model = models.resnet50(pretrained=True)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model


def preprocess_image(image_path, x, y, width, height):
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


def extract_features(image_path, model, x, y, width, height):
    image = preprocess_image(image_path, x, y, width, height)
    with torch.no_grad():
        features = model(image)
    return features.squeeze()


# todo: compare the embedding of one picture to the other
# todo: then use the cosine similarity and the iou to the cost matrix
# todo: color diagram ?
def hungarian(frame_detections: pd.DataFrame, active_tracks: list[Track], currentFrame: Frame,
              cost_matrix: np.ndarray, sigma_iou: float = 0.4, kalmanFilter: bool = False) -> None:
    for track_idx, (track) in enumerate(active_tracks):
        for det_idx, (_, det) in enumerate(frame_detections.iterrows()):
            det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]
            embedding = extract_features(f'ADL-Rundle-6/img1/{currentFrame.frameNumber:06}.jpg', model,
                                         *det_box)
            if track.prediction is not None:  # ONLY set if Kalman filter is TRUE
                iou = compute_iou(det_box, track.prediction)
            else:
                iou = compute_iou(det_box, track.detection)
            similarity = cosine_similarity([embedding], [])  # todo add the previous embedding of the previous box
            cost_matrix[track_idx, det_idx] = 1 - iou

    # Apply the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    for track_idx, det_idx in zip(row_indices, col_indices):
        # Get the corresponding IoU score from the cost matrix
        iou_score = 1 - cost_matrix[track_idx, det_idx]
        det_idx = list(col_indices).index(det_idx)  # Find the detection index
        det = frame_detections.iloc[det_idx]
        det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]

        if iou_score >= sigma_iou:
            # If IoU is above the threshold, update the track with the new detection
            track_id = active_tracks[track_idx].id
            currentFrame.update_track(track_id=track_id, detection=det_box)
        else:
            currentFrame.add_track(detection=det_box, kalmanFilter=kalmanFilter)
