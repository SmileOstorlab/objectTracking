import numpy as np
from KalmanFilter import KalmanFilter
from Detector import detect
import cv2


# Assuming the KalmanFilter class is defined as previously discussed
# Assuming the detect function is defined as provided

def main():
    # Initialize the Kalman filter
    kf = KalmanFilter(dt=0.1, u_x=0, u_y=0, std_acc=1, std_meas_x=0.1, std_meas_y=0.1)

    cap = cv2.VideoCapture('randomball.avi')

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video.")
        return

    # Define a colors for the circle and rectangle
    circle_color = (0, 255, 0)  # Green
    rect_color = (0, 0, 255)  # Red
    predicted_rect_color = (255, 0, 0)  # Blue

    # Initialize a list to store the trajectory points
    trajectory_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Detect circles in the frame
        centers = detect(frame)

        predicted_state, _ = kf.predict()
        cv2.rectangle(frame,
                      (int(predicted_state[0][0]) - 10, int(predicted_state[1][0]) - 10),
                      (int(predicted_state[0][0]) + 10, int(predicted_state[1][0]) + 10),
                      predicted_rect_color, 2)

        # If a centroid is detected, track it
        if centers:
            # Use the first detected center for simplicity
            measurement = centers[0].flatten()
            kf.predict()  # Predict the next state
            estimated_state, _ = kf.update(measurement)  # Update with the actual measurement
            # Predict the next state

            # Draw the tracking results
            # Circle for detected object
            cv2.circle(frame, tuple(measurement.astype(int)), 3, circle_color, -1)
            # Rectangle for the estimated object position
            cv2.rectangle(frame,
                          (int(estimated_state[0][0]) - 10, int(estimated_state[1][0]) - 10),
                          (int(estimated_state[0][0]) + 10, int(estimated_state[1][0]) + 10),
                          rect_color, 2)
            trajectory_points.append(
                (int(estimated_state[0][0]), int(estimated_state[1][0]))
            )

        if len(trajectory_points) > 1:
            for i in range(len(trajectory_points) - 1):
                cv2.line(frame, trajectory_points[i], trajectory_points[i + 1], (0, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
