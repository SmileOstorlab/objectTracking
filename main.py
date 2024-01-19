import subprocess

from TracksHandler import computeTracks
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
    input_file_name = "time.log"
    output_file_name = "benchmark.txt"

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


sigma_iou = 0.5

# frames = computeTracks(sigma_iou=sigma_iou, Hungarian=False, kalmanFilter=False)
# create_csv(frames=frames,
#            csv_filename='benchmark/data/trackers/mot_challenge/MOT15-train/MPNTrack/data/ADL-Rundle-6.txt')
# get_metrics()
# write_lines(f'greedy - {sigma_iou}')
# visualisation(frames=frames)
# create_video(output_name='greedy.mp4', frame_rate=10)

# frames = computeTracks(sigma_iou=sigma_iou, Hungarian=True, kalmanFilter=False)
# create_csv(frames=frames,
#            csv_filename='benchmark/data/trackers/mot_challenge/MOT15-train/MPNTrack/data/ADL-Rundle-6.txt')
# get_metrics()
# write_lines(f'Hungarian - {sigma_iou}')
# visualisation(frames=frames)
# create_video(output_name='hungarian.mp4', frame_rate=10)

frames = computeTracks(sigma_iou=sigma_iou, Hungarian=False, kalmanFilter=True)
create_csv(frames=frames,
           csv_filename='benchmark/data/trackers/mot_challenge/MOT15-train/MPNTrack/data/ADL-Rundle-6.txt')
# get_metrics()
# write_lines(f'KalmanFilter - {sigma_iou}')
visualisation(frames=frames)
# create_video(output_name='greedy_kalman.mp4')

# frames = computeTracks(sigma_iou=sigma_iou, Hungarian=True, kalmanFilter=True)
# frames = computeTracks(sigma_iou=sigma_iou, Hungarian=True, kalmanFilter=False)
# visualisation(frames=frames)
# create_video(output_name='hungarian_kalman.mp4')
