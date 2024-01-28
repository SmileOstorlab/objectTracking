import pandas as pd

from src.Comparaison_Metrics import compute_iou
from Track import Track, Frame


def greedy_matching(frame_detections: pd.Series, active_tracks: list[Track], currentFrame: Frame,
                    sigma_iou: float = 0.4, kalmanFilter: bool = False) -> None:
    """
    Matches frame detections to active tracks using a greedy algorithm based on IoU (Intersection over Union) metric.

    This function iterates over each detection in the current frame and finds the active track with the highest IoU.
    If the highest IoU exceeds a defined threshold (sigma_iou), the detection is associated with that track. Otherwise,
    a new track is created for the detection. The function optionally uses Kalman filter predictions for calculating IoU.

    Args:
        frame_detections (pd.Series): A pandas Series containing detections for the current frame. Each detection should
                                      have 'bb_left', 'bb_top', 'bb_width', and 'bb_height' attributes.
        active_tracks (list[Track]): A list of currently active tracks.
        currentFrame (Frame): The current frame object to which detections and tracks are being matched.
        sigma_iou (float, optional): The IoU threshold for matching detections to tracks. Defaults to 0.4.
        kalmanFilter (bool, optional): A flag to determine whether to use Kalman filter predictions for IoU calculations.
                                       Defaults to False.

    Returns:
        None: The function updates the currentFrame object but does not return any value.
    """
    for _, det in frame_detections.iterrows():
        det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]
        best_iou = 0
        best_track = None

        for track in active_tracks:
            if track.prediction is not None:  # ONLY set if Kalman filter is TRUE
                iou = compute_iou(det_box, track.prediction)
            else:
                iou = compute_iou(det_box, track.detection)

            if iou > best_iou:
                best_iou = iou
                best_track = track.id

        # If the best IoU is above the threshold, update the track with the new detection
        if best_iou >= sigma_iou:
            currentFrame.update_track(track_id=best_track, detection=det_box)
        else:
            currentFrame.add_track(detection=det_box, use_kalmanFilter=kalmanFilter)
