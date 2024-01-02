from TracksHandler import computeTracks
from Visualisation import visualisation, create_video


sigma_iou = 0.7
frames = computeTracks(sigma_iou=sigma_iou, Hungarian=False, kermanFilter=False)
visualisation(frames=frames)
create_video(output_name='greedy.mp4')
frames = computeTracks(sigma_iou=sigma_iou, Hungarian=True, kermanFilter=False)
visualisation(frames=frames)
create_video(output_name='hungarian.mp4')

