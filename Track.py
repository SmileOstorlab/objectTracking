from typing import Optional

from tp1.KalmanFilter import KalmanFilter


class Track:
    def __init__(self, id: int, detection: list[int], kalmanFilter: bool = False) -> None:
        self.id = id
        self.detection = detection
        self.kalmanFilter = None
        self.center = None
        self.prediction = None
        if kalmanFilter:
            self.kalmanFilter = KalmanFilter(dt=0.1, u_x=0, u_y=0, std_acc=1, std_meas_x=0.1, std_meas_y=0.1)

    def update(self, detection: [list[int]]):
        if self.kalmanFilter is not None:
            center_x = detection[0] + (detection[2] / 2)
            center_y = detection[1] + (detection[3] / 2)
            self.center = [center_x, center_y]
            res, _ = self.kalmanFilter.update(self.center)
            new_x, new_y = res[0][0] - (detection[2] / 2), res[1][0] - (detection[3] / 2)
            self.prediction = [new_x, new_y, detection[2], detection[3]]

            self.detection = detection

        else:
            self.detection = detection


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
        self.tracks: dict[int, Track] = {} if matched_tracks is None else {track.id: track for track in matched_tracks}

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

    def add_track(self, detection: list[int], kalmanFilter: bool = False) -> None:
        """Add a new track to the current frame

        :param detection: box to give the track
        :param kalmanFilter: boolean value to use a kalman filter
        :return: None
        """
        track_id = self.idManager.generate_new_id()
        self.add_active_track(track_id=track_id)
        self.tracks[track_id] = Track(id=track_id, detection=detection, kalmanFilter=kalmanFilter)

    def update_track(self, track_id: int, detection: list[int]) -> None:
        """Update the given track with its new detection

        :param track_id: id of the track to update
        :param detection: detection box
        :return: None
        """
        self.add_active_track(track_id=track_id)
        self.tracks[track_id].update(detection=detection)
