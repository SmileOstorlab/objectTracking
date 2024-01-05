import numpy as np
import pandas as pd

from Iou import compute_iou
from PreProcessing import preprocess
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from time import sleep
from typing import Optional
from tp1.KalmanFilter import KalmanFilter


class Track:
    def __init__(self, id: int, detection: list[int], kalmanFilter: bool = False) -> None:
        self.id = id
        self.detection = detection
        self.kalmanFilter = None
        if kalmanFilter:
            self.kalmanFilter = KalmanFilter(dt=0.1, u_x=0, u_y=0, std_acc=1, std_meas_x=0.1, std_meas_y=0.1)

    def update(self, detection: [list[int]]):
        if self.kalmanFilter is not None:
            print(f'detection before prediction: {detection}')
            self.kalmanFilter.update(detection)
            self.detection = self.kalmanFilter.predict()
            print(f'prediction after prediction: {self.detection}')
        else:
            self.detection = detection


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

    def add_active_track(self, track_id: int) -> None:
        self.active_ids.add(track_id)

    def get_active_tracks(self) -> set[int]:
        return self.active_ids


class Frame:
    def __init__(self, frameNumber: int, matched_tracks: Optional[dict[int, Track]] = None):
        self.idManager = IDManager()
        self.frameNumber = frameNumber
        self.tracks: dict[int, Track] = {} if matched_tracks is None else matched_tracks  # list of pair id/Track

    def get_active_track(self) -> list[Track]:
        ret = []
        active_tracks = self.idManager.get_active_tracks()
        for track_id in active_tracks:
            ret.append(self.tracks[track_id])
        return ret

    def add_track(self, detection: list[int], kalmanFilter: bool = False) -> None:
        track_id = self.idManager.generate_new_id()
        self.tracks[track_id] = Track(id=track_id, detection=detection, kalmanFilter=kalmanFilter)

    def update_track(self, track_id: int, detection: list[int]) -> None:
        self.idManager.add_active_track(track_id=track_id)
        self.tracks[track_id].update(detection=detection)


def hungarian(frame_detections: pd.DataFrame, frames: list[Frame], currentFrame: Frame,
              cost_matrix: np.ndarray, sigma_iou: float = 0.4, kalmanFilter: bool = False) -> None:
    for track_idx, (track_id, track) in enumerate(frames[-1].tracks.items()):
        for det_idx, (_, det) in enumerate(frame_detections.iterrows()):
            det_box = [det['bb_left'], det['bb_top'], det['bb_left'] + det['bb_width'],
                       det['bb_top'] + det['bb_height']]
            iou = compute_iou(det_box, track.detection)
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
            track_id = list(frames[-1].tracks.keys())[track_idx]
            currentFrame.update_track(track_id=track_id, detection=det_box)
        else:
            currentFrame.add_track(detection=det_box, kalmanFilter=kalmanFilter)


def greedy(frame_detections: pd.DataFrame, frames: list[Frame], currentFrame: Frame,
           sigma_iou: float = 0.4, kalmanFilter: bool = False) -> None:
    for _, det in frame_detections.iterrows():
        det_box = [det['bb_left'], det['bb_top'], det['bb_left'] + det['bb_width'],
                   det['bb_top'] + det['bb_height']]
        best_iou = 0
        best_track = None

        for curr_id, track in frames[-1].tracks.items():
            iou = compute_iou(det_box, track.detection)

            if iou > best_iou:
                best_iou = iou
                best_track = curr_id

        # If the best IoU is above the threshold, update the track with the new detection
        if best_iou >= sigma_iou:
            currentFrame.update_track(track_id=best_track, detection=det_box)
        else:
            currentFrame.add_track(detection=det_box, kalmanFilter=kalmanFilter)


def computeTracks(sigma_iou: float = 0.4, Hungarian: bool = False, kalmanFilter: bool = False) -> list[Frame]:
    df = preprocess()
    frames: list[Frame] = []
    unique_frames = df['frame'].unique()
    progress_bar = tqdm(total=len(unique_frames), desc="Compute box")

    # Step 3: Associate detections to tracks
    for frame_number in unique_frames:
        frame_detections = df[df['frame'] == frame_number]
        progress_bar.update(1)

        if len(frames) == 0:
            currentFrame = Frame(frame_number)
        else:
            currentFrame = Frame(frame_number, matched_tracks=frames[-1].tracks)

        if len(frames) != 0:
            if not Hungarian:
                greedy(sigma_iou=sigma_iou, frame_detections=frame_detections, frames=frames, currentFrame=currentFrame,
                       kalmanFilter=kalmanFilter)
            else:
                num_tracks = len(frames[-1].tracks) if frames else 0
                num_detections = len(frame_detections)
                cost_matrix = np.ones((num_tracks, num_detections))

                hungarian(sigma_iou=sigma_iou, frame_detections=frame_detections, frames=frames,
                          currentFrame=currentFrame, cost_matrix=cost_matrix, kalmanFilter=kalmanFilter)

        else:  # if no track, add all the boxes to new tracks
            for _, det in frame_detections.iterrows():
                det_box = [det['bb_left'], det['bb_top'], det['bb_left'] + det['bb_width'],
                           det['bb_top'] + det['bb_height']]
                currentFrame.add_track(detection=det_box, kalmanFilter=kalmanFilter)

        frames.append(currentFrame)

    progress_bar.close()
    sleep(0.5)  # for good io...
    return frames
