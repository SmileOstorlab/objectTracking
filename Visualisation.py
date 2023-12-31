import os
import cv2

from typing import Any
from TracksHandler import computeTracks


def visualisation(tracks: list[dict[str, Any]]) -> None:
    images_directory =  'ADL-Rundle-6/img1'
    for image_file in os.listdir(images_directory):
        if not image_file.endswith('.jpg'):
            continue

        image_path = os.path.join(images_directory, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read {image_file}")
            continue

        frame_number = int(image_file.split('.')[0])  # Assumes filenames like '1.jpg', '2.jpg', etc.

        for track in tracks:
            current_id = track['id']
            for box_info in track['detections']:
                if box_info['frame'] == frame_number:
                    # Draw a green bounding box on the image
                    x, y, w, h = box_info['bb_left'], box_info['bb_top'], box_info['bb_width'], box_info['bb_height']
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Put the ID on the image
                    cv2.putText(image, f"ID: {current_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the modified image with boxes and IDs
        output_image_path = os.path.join('img2', f"{frame_number:06}.jpg")
        cv2.imwrite(output_image_path, image)

    print("Images with bounding boxes and IDs have been saved.")


def create_video() -> None:
    # Directory where your JPEG frames are stored
    frames_directory = 'img2'
    frame_files = [f for f in os.listdir(frames_directory) if f.endswith('.jpg')]
    frame_files.sort()

    output_video_filename = 'output_video.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs like 'XVID' or 'MJPG'
    frame_rate = 30  # Adjust the frame rate as needed
    frame_size = (1920, 1080)  # Set the width and height of frames as needed
    out = cv2.VideoWriter(output_video_filename, fourcc, frame_rate, frame_size)

    # Loop through the frame files and add them to the video
    for frame_file in frame_files:
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

    print(f"Video '{output_video_filename}' has been created.")