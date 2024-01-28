import torch
import os

from typing import Optional, Any
from copy import deepcopy

from tp1.KalmanFilter import KalmanFilter
from Comparaison_Metrics import extract_features


def get_image_path(frame_number: int) -> str:
    base_path = os.environ.get('OBJECT_TRACKING_PATH')
    return os.path.join(base_path, f'ADL-Rundle-6/img1/{frame_number:06}.jpg')


class Track:
    """
    Represents a track in object tracking, holding detection, prediction, and state information.

    This class is used to manage the state of an individual track in an object tracking system. It can optionally
    incorporate a Kalman filter for prediction updates and can compute feature embeddings based on a given model.

    Attributes:
       id (int): The unique identifier for the track.
       detection (list[int]): The bounding box detection for the track.
       center (Optional[list[float]]): The center point of the track's bounding box.
       prediction (Optional[list[int]]): The predicted bounding box using a Kalman filter, if enabled.
       current_embedding (Optional[torch.Tensor]): The current feature embedding of the object in the track.
       kalmanFilter (Optional[KalmanFilter]): A Kalman filter instance for the track, if enabled.

    Args:
       id (int): The unique identifier for the track.
       detection (list[int]): The initial bounding box detection for the track.
       use_kalmanFilter (bool, optional): Flag to enable the use of a Kalman filter. Defaults to False.
       model (Optional[Any], optional): The model used for feature extraction. Defaults to None.
       frame_number (int, optional): The frame number for initial embedding computation. Defaults to -1.
    """

    def __init__(self, id: int, detection: list[int], use_kalmanFilter: bool = False,
                 model: Optional[Any] = None, frame_number: int = -1) -> None:
        self.id: int = id
        self.detection: list[int] = detection
        self.center: Optional[list[float]] = None
        self.prediction: Optional[list[int]] = None
        self.current_embedding: Optional[torch.Tensor] = None
        if use_kalmanFilter:
            self.kalmanFilter = KalmanFilter(dt=0.1, u_x=0, u_y=0, std_acc=1, std_meas_x=0.1, std_meas_y=0.1)
        else:
            self.kalmanFilter: Optional[KalmanFilter] = None
        if model is not None:
            self.set_current_embedding(frame_number=frame_number)

    def update(self, detection: list[int], frame_number: int = -1, model: Optional[Any] = None) -> None:
        """
        Updates the track with new detection data and recalculates the state.

        This method updates the track's detection and optionally recalculates the prediction using a Kalman filter.
        If a feature extraction model is provided, it also updates the current feature embedding of the track.

        Args:
          detection (list[int]): The updated bounding box detection for the track.
          frame_number (int, optional): The frame number for updating the embedding. Defaults to -1.
          model (Optional[Any], optional): The model used for feature extraction. Defaults to None.
        """
        if self.kalmanFilter is not None:
            center_x = detection[0] + (detection[2] / 2)
            center_y = detection[1] + (detection[3] / 2)
            self.center = [center_x, center_y]
            res, _ = self.kalmanFilter.update(self.center)
            new_x, new_y = res[0][0] - (detection[2] / 2), res[1][0] - (detection[3] / 2)
            self.prediction = [new_x, new_y, detection[2], detection[3]]

        self.detection = detection

        if self.current_embedding is not None:
            self.set_current_embedding(frame_number=frame_number)

    def set_current_embedding(self, frame_number: int) -> None:
        """
        Sets the current feature embedding of the track based on the provided model.

        This method calculates the feature embedding for the current or predicted bounding box of the track.

        Args:
            frame_number (int): The frame number for feature extraction.
        """
        if self.current_embedding is None:
            image_path = get_image_path(frame_number=frame_number)
            if self.prediction is not None:
                x, y, width, height = self.prediction
                self.current_embedding = extract_features(image_path=image_path, x=x, y=y, width=width,
                                                          height=height).cpu().numpy()
            else:
                x, y, width, height = self.detection
                self.current_embedding = extract_features(image_path=image_path, x=x, y=y, width=width,
                                                          height=height).cpu().numpy()


class IDManager:
    def __init__(self):
        self.next_id = 0

    def generate_new_id(self) -> int:
        """Generate a new ID and mark it as active."""
        new_id = self.next_id
        self.next_id += 1
        return new_id


class Frame:
    def __init__(self, frameNumber: int, idManager: IDManager, matched_tracks: Optional[list[Track]] = None) -> None:
        """Class to represent frames

        :param frameNumber: frame number
        :param matched_tracks: active track in the previous frame
        """
        self.idManager = idManager
        self.frameNumber = frameNumber
        self.active_track = set()
        self.tracks: dict[int, Track] = {} if matched_tracks is None else {track.id: deepcopy(track) for track in
                                                                           matched_tracks}

    def add_active_track(self, track_id: int) -> None:
        self.active_track.add(track_id)

    def get_active_tracks(self) -> set[int]:
        return self.active_track

    def get_active_track(self) -> list[Track]:
        """return all the active tracks on that frame

        :return: list of Track objects
        """
        ret = []
        active_tracks = self.get_active_tracks()
        for track_id in active_tracks:
            ret.append(self.tracks[track_id])
        return ret

    def add_track(self, detection: list[int], use_kalmanFilter: bool = False,
                  model: Optional[Any] = None, frame_number: int = -1) -> None:
        """Add a new track to the current frame

        :param detection: box to give the track
        :param use_kalmanFilter: boolean value to use a kalman filter
        :param frame_number: frame number to get the path of the picture
        :param model: NN model to get the embedding from
        :return: None
        """
        track_id = self.idManager.generate_new_id()
        self.add_active_track(track_id=track_id)
        self.tracks[track_id] = Track(id=track_id, detection=detection, use_kalmanFilter=use_kalmanFilter,
                                      model=model, frame_number=frame_number)

    def update_track(self, track_id: int, detection: list[int]) -> None:
        """Update the given track with its new detection

        :param track_id: id of the track to update
        :param detection: detection box
        :return: None
        """
        self.add_active_track(track_id=track_id)
        self.tracks[track_id].update(detection=detection)
