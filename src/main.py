import subprocess
import os

from TracksHandler import computeTracks, Methode
from Visualisation import visualisation, create_video, create_csv


def get_metrics():
    command = [
        "/home/smile/Documents/object_tracking/benchmark/venv/bin/python",
        "/home/smile/Documents/object_tracking/benchmark/scripts/run_mot_challenge.py",
        "--BENCHMARK", "MOT15",
        "--SPLIT_TO_EVAL", "train",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--NUM_PARALLEL_CORES", "1"
    ]

    working_directory = "/home/smile/Documents/object_tracking/benchmark"

    output_file_path = "/home/smile/Documents/object_tracking/time.log"

    with open(output_file_path, "w") as output_file:
        result = subprocess.run(command, stdout=output_file, stderr=subprocess.PIPE, text=True, cwd=working_directory)

    if result.returncode == 0:
        print("Command executed successfully.")
    else:
        print("Command failed with error:")
        print(result.stderr)


def write_lines(version_name: str):
    input_file_name = "/home/smile/Documents/object_tracking/time.log"
    output_file_name = "/home/smile/Documents/object_tracking/benchmark.txt"

    with open(input_file_name, "r") as input_file, open(output_file_name, "a") as output_file:
        output_file.write(f'\t\t{version_name}\n')
        # Loop through each line in the input file
        for line in input_file:
            # Check if the line starts with "ADL-Rundle-6"
            if line.startswith("ADL-Rundle-6"):
                # Append the line to the output file
                output_file.write(line)

    input_file.close()
    output_file.close()


# sigma_iou = 0.5
# for sigma_iou in [0.15, 0.5, 0.8, 0.9]:
# for sigma_iou in [0.9]:
#     for methode in Methode:
#         frames = computeTracks(sigma_iou=sigma_iou, methode=methode, kalman_filter=False)
#         create_csv(frames=frames,
#                    csv_filename='benchmark/data/trackers/mot_challenge/MOT15-train/MPNTrack/data/ADL-Rundle-6.txt')
#         get_metrics()
#         write_lines(f'\t\t{methode} -[σ = {sigma_iou}]')
#
#         frames = computeTracks(sigma_iou=sigma_iou, methode=methode, kalman_filter=True)
#         create_csv(frames=frames,
#                    csv_filename='benchmark/data/trackers/mot_challenge/MOT15-train/MPNTrack/data/ADL-Rundle-6.txt')
#         get_metrics()
#         write_lines(f'\t\t{methode} - Kalman -[σ = {sigma_iou}]')

# visualisation(frames=frames)
# create_video(output_name='greedy.mp4', frame_rate=10)


os.environ['OBJECT_TRACKING_PATH'] = '/home/smile/Documents/object_tracking'


base_path = os.environ.get('OBJECT_TRACKING_PATH')
csv_path = os.path.join(base_path, 'benchmark/data/trackers/mot_challenge/MOT15-train/MPNTrack/data/ADL-Rundle-6.txt')

for sigma_iou in [0.8]:
    for kalman_filter in [False]:
        methode = Methode.YOLO
        frames = computeTracks(methode=methode, sigma_iou=sigma_iou, kalman_filter=kalman_filter)
        create_csv(frames=frames, csv_filename=csv_path)
        get_metrics()
        write_lines(f'\t\t{methode} - [Kalman = {kalman_filter}] -[σ = {sigma_iou}]')
        visualisation(frames)

        video_path = os.path.join(base_path, f'{methode}-{sigma_iou}-KalmanFilter{kalman_filter}.mp4')

        # create_video(output_name=video_path, frame_rate=30)
