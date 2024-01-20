import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Optional

from Comparaison_Metrics import compute_iou, extract_features
from Track import Track, Frame
from hungarian import hungarian_algorithm


def compute_combined_score(iou, image_embedding, alpha):
    return alpha * image_embedding + (1 - alpha) * iou


def hungarian_improved(frame_detections: pd.Series, active_tracks: list[Track], currentFrame: Frame,
                       cost_matrix: np.ndarray, threshold: float = 0.4, kalmanFilter: bool = False,
                       model: Optional[Any] = None) -> None:
    for track_idx, (track) in enumerate(active_tracks):
        for det_idx, (_, det) in enumerate(frame_detections.iterrows()):
            det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]

            embedding = extract_features(f'ADL-Rundle-6/img1/{currentFrame.frameNumber:06}.jpg', model,
                                         *det_box).cpu().numpy()
            #  !! the result is only computed once, then is it cached

            if track.prediction is not None:  # ONLY set if Kalman filter is TRUE
                iou = compute_iou(det_box, track.prediction)
            else:
                iou = compute_iou(det_box, track.detection)

            similarity = cosine_similarity([embedding], [track.current_embedding])

            alpha = 0.3  # coefficient that determine the importance of the iou in the score

            combined_score = compute_combined_score(iou, similarity, alpha)
            # print(f'combined score {combined_score}, iou: {iou}, similarity: {similarity}')
            cost_matrix[track_idx, det_idx] = 1 - combined_score

    hungarian_algorithm(cost_matrix=cost_matrix, currentFrame=currentFrame, active_tracks=active_tracks,
                        frame_detections=frame_detections, threshold=threshold, kalmanFilter=kalmanFilter,
                        model=model, frame_number=currentFrame.frameNumber)
