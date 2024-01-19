import numpy as np


def get_rectangle_coordinate(det: list[int]) -> list[int]:
    return [det[0], det[1], det[0] + det[2],
            det[1] + det[3]]


def compute_iou(boxA: list[int], boxB: list[int]) -> float:
    boxA, boxB = get_rectangle_coordinate(boxA), get_rectangle_coordinate(boxB)
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
