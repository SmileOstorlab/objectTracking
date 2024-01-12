from TracksHandler import computeTracks
from Visualisation import visualisation, create_video


sigma_iou = 0.7
frames = computeTracks(sigma_iou=sigma_iou, Hungarian=False, kalmanFilter=False)
visualisation(frames=frames)
create_video(output_name='greedy.mp4', frame_rate=10)

frames = computeTracks(sigma_iou=sigma_iou, Hungarian=True, kalmanFilter=False)
visualisation(frames=frames)
create_video(output_name='hungarian.mp4', frame_rate=10)

# frames = computeTracks(sigma_iou=sigma_iou, Hungarian=False, kalmanFilter=True)
# visualisation(frames=frames)
# create_video(output_name='greedy_kalman.mp4')

# frames = computeTracks(sigma_iou=sigma_iou, Hungarian=True, kalmanFilter=True)
# frames = computeTracks(sigma_iou=sigma_iou, Hungarian=True, kalmanFilter=False)
# visualisation(frames=frames)
# create_video(output_name='hungarian_kalman.mp4')

