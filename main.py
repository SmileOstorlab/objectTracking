from TracksHandler import computeTracks
from Visualisation import visualisation, create_video

frames = computeTracks(sigma_iou=0.4, Hungarian=False, kermanFilter=False)
visualisation(frames=frames)
create_video()

