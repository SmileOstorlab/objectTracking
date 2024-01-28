import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from typing import Optional, Any

from Comparaison_Metrics import compute_iou
from Track import Track, Frame


def hungarian(frame_detections: pd.Series, active_tracks: list[Track], currentFrame: Frame,
              cost_matrix: np.ndarray, sigma_iou: float = 0.4, kalmanFilter: bool = False) -> None:
    for track_idx, track in enumerate(active_tracks):
        for det_idx, (_, det) in enumerate(frame_detections.iterrows()):
            det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]
            if track.prediction is not None:  # ONLY set if Kalman filter is TRUE
                iou = compute_iou(det_box, track.prediction)
            else:
                iou = compute_iou(det_box, track.detection)
            cost_matrix[track_idx, det_idx] = 1 - iou

    # Apply the Hungarian algorithm
    hungarian_algorithm(cost_matrix=cost_matrix, currentFrame=currentFrame, active_tracks=active_tracks,
                        frame_detections=frame_detections, threshold=sigma_iou, kalmanFilter=kalmanFilter)


def hungarian_algorithm(cost_matrix, currentFrame: Frame, active_tracks: list[Track],
                        frame_detections: pd.Series | pd.DataFrame, threshold: float = 0.4, kalmanFilter: bool = False,
                        model: Optional[Any] = None, frame_number: int = -1) -> None:
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    for track_idx, det_idx in zip(row_indices, col_indices):
        # Get the corresponding IoU score from the cost matrix
        iou_score = 1 - cost_matrix[track_idx, det_idx]
        det = frame_detections.iloc[det_idx]
        det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]
        # print(det_box)

        if iou_score >= threshold:
            # If IoU is above the threshold, update the track with the new detection
            track_id = active_tracks[track_idx].id
            currentFrame.update_track(track_id=track_id, detection=det_box)
        else:
            currentFrame.add_track(detection=det_box, use_kalmanFilter=kalmanFilter, model=model,
                                   frame_number=frame_number)

    # Handle unmatched detections
    unmatched_detections = set(range(len(frame_detections))) - set(col_indices)
    for det_idx in unmatched_detections:
        det = frame_detections.iloc[det_idx]
        det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]
        currentFrame.add_track(detection=det_box, use_kalmanFilter=kalmanFilter, model=model,
                               frame_number=frame_number)
