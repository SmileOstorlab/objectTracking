import numpy as np
import pandas as pd

from Iou import compute_iou
from PreProcessing import preprocess
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from time import sleep


class Frame:
    def __init__(self, frameNumber: int):
        self.frameNumber = frameNumber
        self.ids: dict[int, list[int]] = {}  # list of pair id/box

    def add_detection(self, id: int, detection: list[int]) -> None:
        self.ids[id] = detection


class IDManager:
    def __init__(self):
        self.next_id = 0
        self.active_ids = set()

    def generate_new_id(self) -> int:
        """Generate a new ID and mark it as active."""
        new_id = self.next_id
        self.next_id += 1
        self.active_ids.add(new_id)
        return new_id


def hungarian(frame_detections: pd.DataFrame, frames: list[Frame], currentFrame: Frame, idManager: IDManager,
              cost_matrix: np.ndarray, sigma_iou: float = 0.4) -> None:

    for track_idx, (curr_id, old_box) in enumerate(frames[-1].ids.items()):
        for det_idx, (_, det) in enumerate(frame_detections.iterrows()):
            det_box = [det['bb_left'], det['bb_top'], det['bb_left'] + det['bb_width'],
                       det['bb_top'] + det['bb_height']]
            iou = compute_iou(det_box, old_box)
            cost_matrix[track_idx, det_idx] = 1 - iou

    # Apply the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    for track_idx, det_idx in zip(row_indices, col_indices):
        # Get the corresponding IoU score from the cost matrix
        iou_score = 1 - cost_matrix[track_idx, det_idx]
        det_idx = list(col_indices).index(det_idx)  # Find the detection index
        det = frame_detections.iloc[det_idx]
        det_box = [det['bb_left'], det['bb_top'], det['bb_left'] + det['bb_width'],
                   det['bb_top'] + det['bb_height']]

        if iou_score >= sigma_iou:
            # If IoU is above the threshold, update the track with the new detection
            track_id = list(frames[-1].ids.keys())[track_idx]
            currentFrame.add_detection(id=track_id, detection=det_box)
        else:
            currentFrame.add_detection(id=idManager.generate_new_id(), detection=det_box)


def greedy(frame_detections: pd.DataFrame, frames: list[Frame], currentFrame: Frame, idManager: IDManager,
           sigma_iou: float = 0.4) -> None:
    for _, det in frame_detections.iterrows():
        det_box = [det['bb_left'], det['bb_top'], det['bb_left'] + det['bb_width'],
                   det['bb_top'] + det['bb_height']]
        best_iou = 0
        best_track = None

        for curr_id, old_box in frames[-1].ids.items():
            iou = compute_iou(det_box, old_box)

            if iou > best_iou:
                best_iou = iou
                best_track = curr_id

        # If the best IoU is above the threshold, update the track with the new detection
        if best_iou >= sigma_iou:
            currentFrame.add_detection(id=best_track, detection=det_box)
        else:
            currentFrame.add_detection(id=idManager.generate_new_id(), detection=det_box)


def computeTracks(sigma_iou: float = 0.4, Hungarian: bool = False, kermanFilter: bool = False) -> list[Frame]:
    df = preprocess()
    idManager = IDManager()
    frames: list[Frame] = []
    unique_frames = df['frame'].unique()
    progress_bar = tqdm(total=len(unique_frames), desc="Compute box")

    # Step 3: Associate detections to tracks
    for frame_number in unique_frames:
        currentFrame = Frame(frame_number)
        frame_detections = df[df['frame'] == frame_number]
        progress_bar.update(1)

        if len(frames) != 0:
            if not Hungarian:
                greedy(sigma_iou=sigma_iou, frame_detections=frame_detections, frames=frames, currentFrame=currentFrame,
                       idManager=idManager)
            elif kermanFilter:
                raise NotImplemented
            else:
                num_tracks = len(frames[-1].ids) if frames else 0
                num_detections = len(frame_detections)
                cost_matrix = np.ones((num_tracks, num_detections))

                hungarian(sigma_iou=sigma_iou, frame_detections=frame_detections, frames=frames,
                          currentFrame=currentFrame, idManager=idManager, cost_matrix=cost_matrix)

        else:  # if no track, add all the boxes to new tracks
            for _, det in frame_detections.iterrows():
                det_box = [det['bb_left'], det['bb_top'], det['bb_left'] + det['bb_width'],
                           det['bb_top'] + det['bb_height']]
                currentFrame.add_detection(id=idManager.generate_new_id(), detection=det_box)

        frames.append(currentFrame)

    progress_bar.write(f'found {idManager.next_id} total tracks')
    progress_bar.close()
    sleep(0.5)  # for good io...
    return frames
