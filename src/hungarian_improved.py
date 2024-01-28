import numpy as np
import pandas as pd
import os

from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Optional

from Comparaison_Metrics import compute_iou, extract_features
from Track import Track, Frame
from hungarian import hungarian_algorithm


def compute_combined_score(iou, image_embedding, alpha):
    return alpha * image_embedding + (1 - alpha) * iou


def improved_cost_matrix(frame_detections: pd.Series, active_tracks: list[Track], currentFrame: Frame,
                         cost_matrix: np.ndarray, threshold: float = 0.4, kalmanFilter: bool = False,
                         model: Optional[Any] = None) -> None:
    """
    Generates an advanced cost matrix for frame detections and active tracks using both IoU and feature similarity.

    This function computes a cost matrix where each element is a combined score of IoU (Intersection over Union) and
    cosine similarity of feature embeddings. It calculates these scores for each detection in relation to each track and 
    then uses them to populate the cost matrix. The function also applies the Hungarian algorithm to match detections 
    with tracks based on this cost matrix.

    Args:
        frame_detections (pd.Series): A pandas Series containing detections for the current frame, with each detection
                                      having 'bb_left', 'bb_top', 'bb_width', and 'bb_height' attributes.
        active_tracks (list[Track]): A list of currently active tracks.
        currentFrame (Frame): The current frame object to which detections and tracks are being matched.
        cost_matrix (np.ndarray): An array to be filled with the combined scores for each detection-track pair.
        threshold (float, optional): The threshold for considering a detection and track match. Defaults to 0.4.
        kalmanFilter (bool, optional): A flag to determine whether to use Kalman filter predictions for IoU calculations.
                                       Defaults to False.
        model (Optional[Any], optional): An optional model parameter for feature extraction. Defaults to None.

    Returns:
        None: The function updates the cost_matrix and uses it in the Hungarian algorithm for track matching but does not 
        return any value.
    """
    base_path = os.environ.get('OBJECT_TRACKING_PATH')

    for track_idx, (track) in enumerate(active_tracks):
        for det_idx, (_, det) in enumerate(frame_detections.iterrows()):
            x, y, width, height = det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']
            det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]

            image_path = os.path.join(base_path, f'ADL-Rundle-6/img1/{currentFrame.frameNumber:06}.jpg')
            embedding = extract_features(image_path=image_path, x=x, y=y, width=width, height=height).cpu().numpy()
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
