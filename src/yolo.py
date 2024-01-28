import torch
import pandas as pd
import numpy as np

from Comparaison_Metrics import compute_iou
from Track import Track, Frame
from hungarian import hungarian_algorithm


class YOLOv5Singleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._init_model()
        return cls._instance

    @staticmethod
    def _init_model():
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model


def get_pedestrian_detection(image_path: str) -> pd.Series:
    model = YOLOv5Singleton.get_instance()
    results = model(image_path)
    df = results.pandas().xyxy[0]
    df = df[df['class'] == 0]  # class from pedestrian
    return df[df['confidence'] > 0.6]


def build_det_box(det: pd.Series):
    bb_left = det['xmin']
    bb_top = det['ymin']
    bb_width = det['xmax'] - det['xmin']
    bb_height = det['ymax'] - det['ymin']
    # return pd.Series([bb_left, bb_top, bb_width, bb_height], index=['bb_left', 'bb_top', 'bb_width', 'bb_height'])
    return [bb_left, bb_top, bb_width, bb_height]


def yolo_cost_matrix(active_tracks: list[Track], currentFrame: Frame, image_path: str,
                     threshold: float = 0.4, kalmanFilter: bool = False, ) -> None:
    """
    Generates a cost matrix for active tracks and detections obtained from YOLOv5 model.

    This function applies the YOLOv5 model to detect pedestrians in the given image and creates a cost matrix based on
    the Intersection over Union (IoU) between each YOLO detection and each active track. The cost matrix is used to
    associate detections to tracks using the Hungarian algorithm. It supports the use of a Kalman filter for improved
    track prediction accuracy.

    Args:
       active_tracks (list[Track]): A list of currently active tracks.
       currentFrame (Frame): The current frame object to which detections and tracks are being matched.
       image_path (str): The file path of the image to be processed by the YOLOv5 model.
       threshold (float, optional): The IoU threshold for matching detections to tracks. Defaults to 0.4.
       kalmanFilter (bool, optional): A flag to determine whether to use Kalman filter predictions for IoU calculations.
                                      Defaults to False.

    Returns:
       None: The function updates the currentFrame object with matched tracks but does not return any value.
    """

    frame_detections = get_pedestrian_detection(image_path=image_path)
    cost_matrix = np.ones((len(active_tracks), len(frame_detections)))
    detections = []

    for track_idx, track in enumerate(active_tracks):
        for det_idx, (_, det) in enumerate(frame_detections.iterrows()):

            det_box = build_det_box(det=det)
            detections.append(det_box)

            if track.prediction is not None:  # ONLY set if Kalman filter is TRUE
                iou = compute_iou(det_box, track.prediction)
            else:
                iou = compute_iou(det_box, track.detection)
            cost_matrix[track_idx, det_idx] = 1 - iou

    frame_detections = pd.DataFrame(detections, columns=['bb_left', 'bb_top', 'bb_width', 'bb_height']).drop_duplicates()
    # Apply the Hungarian algorithm
    hungarian_algorithm(cost_matrix=cost_matrix, currentFrame=currentFrame, active_tracks=active_tracks,
                        frame_detections=frame_detections, threshold=threshold, kalmanFilter=kalmanFilter)
