import os
import cv2
from tqdm import tqdm

from TracksHandler import Frame


def visualisation(frames: list[Frame]) -> None:
    progress_bar = tqdm(total=len(frames), desc="Draw Boxes")
    images_directory = 'ADL-Rundle-6/img1'

    for frame in frames:
        progress_bar.update(1)
        image_file = f'{frame.frameNumber:06}.jpg'

        image_path = os.path.join(images_directory, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read {image_file}")
        for track in frame.get_active_track():
            x, y, w, h = track.detection
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
            # Put the ID on the image
            cv2.putText(image, f"ID: {track.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the modified image with boxes and IDs
            output_image_path = os.path.join('img2', image_file)
            cv2.imwrite(output_image_path, image)

    progress_bar.close()


def create_video(output_name: str = 'output_video.mp4', frame_rate=30) -> None:
    # Directory where your JPEG frames are stored
    frames_directory = 'img2'
    frame_files = [f for f in os.listdir(frames_directory) if f.endswith('.jpg')]
    frame_files.sort()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs like 'XVID' or 'MJPG'
    frame_rate = 30  # Adjust the frame rate as needed
    frame_size = (1920, 1080)  # Set the width and height of frames as needed
    out = cv2.VideoWriter(output_name, fourcc, frame_rate, frame_size)

    progress_bar = tqdm(total=len(frame_files), desc=f"Compile Frames into video: {output_name}")

    # Loop through the frame files and add them to the video
    for frame_file in frame_files:
        progress_bar.update(1)
        frame_path = os.path.join(frames_directory, frame_file)
        frame = cv2.imread(frame_path)

        # Check if the frame was read successfully
        if frame is None:
            print(f"Error: Could not read {frame_file}")
            continue

        # Write the frame to the video
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    progress_bar.close()
