from TracksHandler import computeTracks
from Visualisation import visualisation

tracks = computeTracks(sigma_iou=0.5, Hungarian=False, kermanFilter=False)
visualisation(tracks=tracks)
