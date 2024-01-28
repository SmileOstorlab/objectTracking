import numpy as np
import os

from tqdm import tqdm
from time import sleep
from enum import Enum

from PreProcessing import preprocess
from Track import IDManager, Frame
from greedy import greedy
from hungarian import hungarian
from hungarian_improved import improved_cost_matrix
from Comparaison_Metrics import ResnetSingleton
from yolo import yolo_cost_matrix


class Methode(Enum):
    GREEDY = 1
    HUNGARIAN = 2
    RESNET = 3
    YOLO = 4


def computeTracks(methode: Methode, kalman_filter: bool = False, sigma_iou: float = 0.4) -> list[Frame]:
    df = preprocess()
    frames: list[Frame] = []
    unique_frames = df['frame'].unique()
    idManager = IDManager()
    progress_bar = tqdm(total=len(unique_frames), desc="Compute box")

    if methode == Methode.RESNET:
        model = ResnetSingleton.get_instance()
    else:
        model = None

    # Step 3: Associate detections to tracks
    for frame_number in unique_frames:
        frame_detections = df[df['frame'] == frame_number]
        progress_bar.update(1)

        if len(frames) == 0:
            currentFrame = Frame(frameNumber=frame_number, idManager=idManager)
        else:
            currentFrame = Frame(frameNumber=frame_number, matched_tracks=frames[-1].get_active_track(),
                                 idManager=idManager)

        if len(frames) != 0:
            if methode == Methode.GREEDY:
                greedy(sigma_iou=sigma_iou, frame_detections=frame_detections, currentFrame=currentFrame,
                       active_tracks=frames[-1].get_active_track(), kalmanFilter=kalman_filter)
            else:
                num_tracks = len(frames[-1].tracks) if frames else 0
                num_detections = len(frame_detections)
                cost_matrix = np.ones((num_tracks, num_detections))
                if methode == Methode.HUNGARIAN:
                    hungarian(sigma_iou=sigma_iou, frame_detections=frame_detections, currentFrame=currentFrame,
                              active_tracks=frames[-1].get_active_track(), cost_matrix=cost_matrix,
                              kalmanFilter=kalman_filter)
                elif methode == Methode.RESNET:
                    improved_cost_matrix(frame_detections=frame_detections, currentFrame=currentFrame,
                                         active_tracks=frames[-1].get_active_track(), cost_matrix=cost_matrix,
                                         threshold=sigma_iou, kalmanFilter=kalman_filter, model=model)
                elif methode == Methode.YOLO:
                    base_path = os.environ.get('OBJECT_TRACKING_PATH')
                    image_path = os.path.join(base_path, f'ADL-Rundle-6/img1/{frame_number:06}.jpg')

                    yolo_cost_matrix(image_path=image_path, currentFrame=currentFrame,
                                     active_tracks=frames[-1].get_active_track(),
                                     threshold=sigma_iou, kalmanFilter=kalman_filter)
        else:  # if no track, add all the boxes to new tracks
            for _, det in frame_detections.iterrows():
                det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]
                currentFrame.add_track(detection=det_box, kalmanFilter=kalman_filter, model=model, frame_number=1)

        frames.append(currentFrame)

    progress_bar.write(f'found {idManager.next_id} total tracks')
    progress_bar.close()
    sleep(0.5)  # for good io...
    return frames
