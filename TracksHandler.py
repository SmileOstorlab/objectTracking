from Iou import compute_iou
from PreProcessing import preprocess
from typing import Any
from tqdm import tqdm

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

        # Compute IoU between each detection and each track
        for _, det in frame_detections.iterrows():
            det_box = [det['bb_left'], det['bb_top'], det['bb_left'] + det['bb_width'],
                       det['bb_top'] + det['bb_height']]
            best_iou = 0
            best_track = None

            if len(frames) != 0:
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

        frames.append(currentFrame)

    progress_bar.close()
    return frames
