import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, std_meas_x, std_meas_y):
        """
        Initialize the Kalman Filter with necessary parameters.

        :param dt: Time step
        :param u_x: Acceleration in the x-direction
        :param u_y: Acceleration in the y-direction
        :param std_acc: Standard deviation of the acceleration
        :param std_meas_x: Standard deviation of the measurement in the x-direction
        :param std_meas_y: Standard deviation of the measurement in the y-direction
        """
        # State transition matrix
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Control input matrix
        self.B = np.array([[0.5 * dt ** 2, 0],
                           [0, 0.5 * dt ** 2],
                           [dt, 0],
                           [0, dt]])

        # Measurement mapping matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process noise covariance matrix
        self.Q = std_acc ** 2 * np.array([[0.25 * dt ** 4, 0, 0.5 * dt ** 3, 0],
                                          [0, 0.25 * dt ** 4, 0, 0.5 * dt ** 3],
                                          [0.5 * dt ** 3, 0, dt ** 2, 0],
                                          [0, 0.5 * dt ** 3, 0, dt ** 2]])

        # Measurement noise covariance matrix
        self.R = np.array([[std_meas_x ** 2, 0],
                           [0, std_meas_y ** 2]])

        # Control input (acceleration)
        self.u = np.array([[u_x], [u_y]])

        # Initial state estimate
        self.x_est = np.zeros((4, 1))

        # Initial covariance matrix
        self.P_est = np.eye(4)

    def predict(self):
        """
        Predict the state estimate and error covariance for the next time step.
        """
        # Predict the state estimate
        self.x_est = np.dot(self.A, self.x_est) + np.dot(self.B, self.u)

        # Predict the error covariance
        self.P_est = np.dot(np.dot(self.A, self.P_est), self.A.T) + self.Q

        return self.x_est, self.P_est

    def update(self, z):
        """
        Update the predicted state estimate and error covariance with the actual measurement.

        :param z: Measurement vector
        """
        # Compute the Kalman Gain
        S = np.dot(self.H, np.dot(self.P_est, self.H.T)) + self.R
        K = np.dot(np.dot(self.P_est, self.H.T), np.linalg.inv(S))
        # Update the state estimate
        z = np.expand_dims(z, axis=1)  # Ensure z is a column vector
        y = z - np.dot(self.H, self.x_est)
        self.x_est = self.x_est + np.dot(K, y)

        # Update the error covariance
        self.P_est = self.P_est - np.dot(np.dot(K, self.H), self.P_est)
        return self.x_est, self.P_est



