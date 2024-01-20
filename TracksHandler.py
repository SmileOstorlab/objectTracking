import numpy as np

from tqdm import tqdm
from time import sleep

from PreProcessing import preprocess
from Track import Track, IDManager, Frame
from greedy import greedy
from hungarian import hungarian


def computeTracks(sigma_iou: float = 0.4, Hungarian: bool = False, kalmanFilter: bool = False) -> list[Frame]:
    df = preprocess()
    frames: list[Frame] = []
    unique_frames = df['frame'].unique()
    idManager = IDManager()
    progress_bar = tqdm(total=len(unique_frames), desc="Compute box")

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
            if not Hungarian:
                greedy(sigma_iou=sigma_iou, frame_detections=frame_detections, currentFrame=currentFrame,
                       active_tracks=frames[-1].get_active_track(), kalmanFilter=kalmanFilter)
            else:
                num_tracks = len(frames[-1].tracks) if frames else 0
                num_detections = len(frame_detections)
                cost_matrix = np.ones((num_tracks, num_detections))

                hungarian(sigma_iou=sigma_iou, frame_detections=frame_detections, currentFrame=currentFrame,
                          active_tracks=frames[-1].get_active_track(), cost_matrix=cost_matrix,
                          kalmanFilter=kalmanFilter)

        else:  # if no track, add all the boxes to new tracks
            for _, det in frame_detections.iterrows():
                det_box = [det['bb_left'], det['bb_top'], det['bb_width'], det['bb_height']]
                currentFrame.add_track(detection=det_box, kalmanFilter=kalmanFilter)

        frames.append(currentFrame)

    progress_bar.write(f'found {idManager.next_id} total tracks')
    progress_bar.close()
    sleep(0.5)  # for good io...
    return frames
