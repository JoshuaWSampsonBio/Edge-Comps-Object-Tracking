import numpy as np

class KF:
    def __init__(self, x, p, q, r, a, h):
        self.x = x
        # state estimate vector (nx1), e.g. 8x1:
        # [x1,y1,x2,y2,vx1,vy1,vx2,vy2]^T

        self.P = p
        # covariance of state estimate (uncertainty)

        self.Q = q
        # process noise covariance: uncertainty from motion model assumptions

        self.R = r
        # measurement noise covariance: uncertainty from detector noise

        self.A = a
        # state transition matrix (motion model)

        self.H = h
        # measurement matrix: extracts observed parts from state

        self.xk_p = x
        self.pk_p = p
        # predicted state/covariance holders


    def predict(self):
        xk_p = np.dot(self.A, self.x)
        # predicted state = A * x

        pk_p = np.dot(np.dot(self.A, self.P), np.transpose(self.A)) + self.Q
        # predicted covariance = A*P*A^T + Q

        self.pk_p = pk_p
        self.xk_p = xk_p
        # store as "prior" for the update step

        return self.xk_p


    def kalman_gain(self):
        S = np.dot(np.dot(self.H, self.pk_p), self.H.T) + self.R
        # innovation covariance: expected uncertainty in measurement space

        Kk = np.dot(np.dot(self.pk_p, self.H.T), np.linalg.inv(S))
        # Kalman gain says how much to trust measurement vs prediction

        return Kk


    def estimate(self, zk):
        K = self.kalman_gain()

        xk = self.xk_p + np.dot(K, (zk - np.dot(self.H, self.xk_p)))
        # correction:
        # zk - H*x_pred is the innovation (measurement residual)

        pk = self.pk_p - np.dot(np.dot(K, self.H), self.pk_p)
        # covariance shrinks if measurement gives information

        return xk, pk


    def update(self, zk):
        self.predict()
        # your update always includes predict first (self-contained)

        self.x, self.P = self.estimate(zk)
        # apply correction

        self.xk_p = self.x
        self.pk_p = self.P
        # sync predicted holders to posterior

        return self.x


    def get_prediction(self):
        return self.predict()
        # convenience wrapper (but note: it ADVANCES state)
