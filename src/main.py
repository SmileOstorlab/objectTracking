import subprocess
import os

from TracksHandler import computeTracks, Method
from Visualisation import visualisation, create_video, create_csv


def get_metrics():
    """
    Executes a benchmark script to evaluate object tracking performance.

    This function runs a benchmarking script for the MOT15 challenge using specific metrics like HOTA, CLEAR, and Identity.
    It executes the script in a subprocess, directing the standard output to a log file and capturing any standard error.
    The function prints a message indicating the success or failure of the command execution.

    Returns:
        None: This function executes a command and prints the result but does not return any value.
    """
    command = [
        "/home/smile/Documents/object_tracking/benchmark/venv/bin/python",
        "/home/smile/Documents/object_tracking/benchmark/scripts/run_mot_challenge.py",
        "--BENCHMARK", "MOT15",
        "--SPLIT_TO_EVAL", "train",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--NUM_PARALLEL_CORES", "1"
    ]

    working_directory = os.path.join(base_path, 'benchmark')

    output_file_path = os.path.join(base_path, 'time.log')

    with open(output_file_path, "w") as output_file:
        result = subprocess.run(command, stdout=output_file, stderr=subprocess.PIPE, text=True, cwd=working_directory)

    if result.returncode == 0:
        print("Command executed successfully.")
    else:
        print("Command failed with error:")
        print(result.stderr)


def write_lines(version_name: str):
    """
    Appends selected lines from a log file to an output file with a version header.

    This function reads lines from an input log file, filters for lines that start with a specific string ("ADL-Rundle-6"),
    and writes these lines to an output file. Each section in the output file starts with the provided version name,
    allowing for easy identification and comparison of different benchmark versions.

    Args:
       version_name (str): The name of the version to be written as a header in the output file.

    Returns:
       None: This function writes to a file but does not return any value.
    """
    input_file_name = os.path.join(base_path, 'time.log')
    output_file_name = os.path.join(base_path, 'benchmark.txt')

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


os.environ['OBJECT_TRACKING_PATH'] = '/home/smile/Documents/object_tracking'

base_path = os.environ.get('OBJECT_TRACKING_PATH')
csv_path = os.path.join(base_path, 'benchmark/data/trackers/mot_challenge/MOT15-train/MPNTrack/data/ADL-Rundle-6.txt')

for method in [Method.GREEDY]:
    for sigma_iou in [0.8]:
        # for sigma in range(1, 20, 1):
        #     sigma_iou = sigma * 0.05
        for kalman_filter in [False, True]:
            print(method, sigma_iou, kalman_filter)
            frames = computeTracks(method=method, sigma_iou=sigma_iou, kalman_filter=kalman_filter)
            create_csv(frames=frames, csv_filename=csv_path)
            get_metrics()
            write_lines(f'\t\t{method} - [Kalman = {kalman_filter}] -[σ = {sigma_iou}]')

            # visualisation(frames)

            # video_path = os.path.join(base_path, f'{methode}-{sigma_iou}-KalmanFilter{kalman_filter}.mp4')
            # create_video(output_name=video_path, frame_rate=30)
