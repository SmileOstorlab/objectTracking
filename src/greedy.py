import pandas as pd

from src.Comparaison_Metrics import compute_iou
from Track import Track, Frame


def greedy(frame_detections: pd.Series, active_tracks: list[Track], currentFrame: Frame,
           sigma_iou: float = 0.4, kalmanFilter: bool = False) -> None:
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
            currentFrame.add_track(detection=det_box, kalmanFilter=kalmanFilter)
