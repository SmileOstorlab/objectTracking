import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from typing import Optional, Any

from Comparaison_Metrics import compute_iou
from Track import Track, Frame


def hungarian_matching(frame_detections: pd.Series, active_tracks: list[Track], currentFrame: Frame,
                       cost_matrix: np.ndarray, sigma_iou: float = 0.4, kalmanFilter: bool = False) -> None:
    """
    Performs matching of frame detections to active tracks using the Hungarian algorithm.

    This function first calculates a cost matrix based on the IoU (Intersection over Union) between each detection and
    each track. It then applies the Hungarian algorithm to find the optimal assignment of detections to tracks.
    The Hungarian algorithm minimizes the total cost of matching detections to tracks.

    Args:
        frame_detections (pd.Series): A pandas Series containing detections for the current frame. Each detection
                                      should have 'bb_left', 'bb_top', 'bb_width', and 'bb_height' attributes.
        active_tracks (list[Track]): A list of currently active tracks.
        currentFrame (Frame): The current frame object to which detections and tracks are being matched.
        cost_matrix (np.ndarray): A pre-initialized cost matrix for storing IoU costs between detections and tracks.
        sigma_iou (float, optional): The IoU threshold for matching detections to tracks. Defaults to 0.4.
        kalmanFilter (bool, optional): A flag to determine whether to use Kalman filter predictions for IoU calculations.
                                       Defaults to False.

    Returns:
        None: The function updates the currentFrame object with matched tracks but does not return any value.
    """
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
    """
    Executes the Hungarian algorithm to assign frame detections to active tracks based on a cost matrix.

    After applying the Hungarian algorithm on the cost matrix, each detection is either assigned to an existing track or
    becomes a new track based on the IoU score calculated and the defined threshold. The algorithm also handles unmatched
    detections by creating new tracks for them.

    Args:
       cost_matrix (np.ndarray): The cost matrix representing the IoU-based costs between each detection and track.
       currentFrame (Frame): The current frame object to which detections and tracks are being matched.
       active_tracks (list[Track]): A list of currently active tracks.
       frame_detections (pd.Series | pd.DataFrame): Detections for the current frame.
       threshold (float, optional): The IoU threshold for matching detections to tracks. Defaults to 0.4.
       kalmanFilter (bool, optional): A flag to determine whether to use Kalman filter predictions for IoU calculations.
                                      Defaults to False.
       model (Optional[Any], optional): An optional model parameter that might be used for track creation. Defaults to None.
       frame_number (int, optional): The current frame number, used optionally in track creation. Defaults to -1.

    Returns:
       None: The function updates the currentFrame object with the assignments but does not return any value.
    """
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
