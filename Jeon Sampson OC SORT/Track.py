# Track.py

import numpy as np
from KF import KF

# new track.py building the required matrices for KF to work
id = 1

class Track:
    def __init__(self,past_observation,timestep):
        ###Give id to Track instance
        global id
        self.id = id
        id += 1

        self.past_observations = [past_observation] # last associated DETECTION bbox: [x1,y1,x2,y2]

        # +++++++++++ tells us current frame index, we can use this to see when predictions were made & how long a track has been unmatched
        self.timestep = timestep
        self.has_filter = False                  # whether a KF is initialized for this track
        self.tracked = True                      # whether it was matched this frame (MOT bookkeeping)
        self.untracked = 0                       # how many frames it has gone unmatched
        self.last_obs_timestep = timestep  # NEW: last frame index where we had a real detection update



        # +++++++++++list of [pred_bbox, timestep] entries, for the predict and get prediction later
        self.prediction_list = []


        # We'll store KF matrices on the track so each track can maintain its own motion model parameters.
        # running KF needs properly shaped matrices, so we can’t just store the raw bbox as
        # the state. Instead, we need to build the A,H,P,Q,R matrices that KF expects.


        self.A = None    # 8x8 state transition matrix
        self.H = None    # 4x8 measurement matrix
        self.P = None    # 8x8 state covariance
        self.Q = None    # 8x8 process noise covariance
        self.R = None    # 4x4 measurement noise covariance




# ++++++++ HELPERS
# making the matrices for the KF to use.
    def _make_kf_mats(self, dt=1.0):
        # dt = timestep between frames (usually 1 for frame-by-frame tracking)
        # State is 8D (adding the velocities for each): [x1,y1,x2,y2,vx1,vy1,vx2,vy2]^T
        # Measurement is 4D (coming from the detections from YOLOX): [x1,y1,x2,y2]^T

        # A: constant-velocity model (positions get updated by velocity*dt)
        # Matrix entry	Meaning
        # A[0,4] = dt	x1 depends on vx1
        # A[1,5] = dt	y1 depends on vy1
        # A[2,6] = dt	x2 depends on vx2
        # A[3,7] = dt	y2 depends on vy2

        A = np.eye(8, dtype=float) #creates the identity matrix, which is the base for A since we want to keep the state the same except for the position updates from velocity
        A[0,4] = dt
        A[1,5] = dt
        A[2,6] = dt
        A[3,7] = dt


        # H: measurement model, we only observe the position components of the 8D state (not velocity)
        # looks at the four position entries of the state and maps them to the measurement space, ignoring the velocity entries.
        H = np.zeros((4,8), dtype=float)
        H[0,0] = 1.0
        H[1,1] = 1.0
        H[2,2] = 1.0
        H[3,3] = 1.0


        # P: initial uncertainty (tunable)
        # 8x8 identity matrix, scaled by a factor (e.g. 10.0) to represent initial uncertainty about the state estimate.
        P = np.eye(8, dtype=float) * 10.0


        # Q: motion/process noise (tunable)
        # makes 8*8 identity matrix, scaled by a factor (e.g. 1.0) to represent
        # how unpredictable the motion is (higher means we trust the motion model
        # less and rely more on measurements).
        Q = np.eye(8, dtype=float) * 1.0


        # R: detection/measurement noise (tunable)
        # makes a 4x4 identity matrix, scaled by a factor (e.g. 10.0) to represent
        # how noisy the detections are (higher means we trust the detections less
        # and rely more on the motion model).
        R = np.eye(4, dtype=float) * 10.0


        # Return all KF matrices
        return A, H, P, Q, R


    def _z_from_bbox(self, bbox):
        # Convert bbox list [x1,y1,x2,y2] to a KF measurement column vector zk (4x1)
        # match dimensions KF expects: [x1,y1,x2,y2] -> [[x1],[y1],[x2],[y2]]
        x1,y1,x2,y2 = bbox
        return np.array([[x1],[y1],[x2],[y2]], dtype=float)


    def _x0_from_bbox(self, bbox):
        # making the initial state vector x0 (8x1) from the initial bbox.
        # We only have position info from the bbox, so we set the velocity entries to 0.
        z = self._z_from_bbox(bbox)
        x0 = np.zeros((8,1), dtype=float)
        x0[0:4] = z
        return x0



# The actual API

    #++++++++++
    def init_filter(self,observation):
        # Initialize a KF for this track from the first observed bbox.
        # made kf consistent with the matrices we defined above and the initial state from the bbox.

        # Create KF matrices
        self.A, self.H, self.P, self.Q, self.R = self._make_kf_mats(dt=1.0)

        # Create initial state x0 from bbox
        x0 = self._x0_from_bbox(observation)

        # Create KF instances and attaching it to a track
        # every tracked object has its own KF instance, which maintains its own internal state and covariance.
        self.kalman_filter = KF(
            # x0​=[x1,y1,x2,y2,0,0,0,0]T
            x=x0,

            # we make copies because KF will update these matrices internally,
            # and we want each track to have its own independent set of matrices,
            # instead of sharing references to the same matrices across tracks.
            p=self.P.copy(),   # copy so each track has its own independent matrices
            q=self.Q.copy(),
            r=self.R.copy(),
            a=self.A.copy(),
            h=self.H.copy()
        )

        # Track now has a filter
        self.has_filter = True

    def check_filter(self):
        # Return True if KF exists for this track
        return self.has_filter


# +++++++++ before, anytime we wanted a prediction, we called get_prediction(),
# which inside KF called predict().
# prediction changes the state because it advances the track forward one time step (one frame)
# we cant keep doing that because then the track would keep moving forward every time we wanted to
# read the prediction, even if it wasn’t matched to a new detection.
# separate the act of running the prediction step (which advances the track forward)
# from the act of just reading the most recent prediction (which should not change the state).
    def predict(self):
        # Run KF prediction step and store predicted bbox.
        # If no filter exists yet, nothing to predict.
        if not self.has_filter:
            return None

        # KF.predict returns an 8x1 predicted state vector
        x_pred = self.kalman_filter.predict()          # (8,1)

        # Convert predicted state -> bbox prediction by taking first 4 entries
        bbox_pred = x_pred[0:4].reshape(-1).tolist()   # [x1,y1,x2,y2]

        # Store prediction for later OCR / debugging
        self.prediction_list.append([bbox_pred, self.timestep, self.id])

        return bbox_pred

# ++++++++++ only retrieves the most recent prediction without running predict() again,
# so it doesn’t advance the track forward.
    def get_prediction(self):
        # Return most recent predicted bbox (or None if predict() hasn’t been called yet)
        if len(self.prediction_list) == 0:
            return None
        return self.prediction_list[-1][0]

    def get_prediction_list(self):
        if len(self.prediction_list) == 0:
            return None
        return self.prediction_list

    def get_past(self,i):
        # Return last associated detection bbox
        return self.past_observations[-i]

    def get_past_list(self):
        # Return last associated detection bbox
        return self.past_observations

    def update(self, observation):
        if self.has_filter:
            zk = self._z_from_bbox(observation)
            self.kalman_filter.update(zk)
            self.past_observations.append(observation)
            self.last_obs_timestep = self.timestep  # NEW


    # !!!!!!! adding the ORU update method to the Track class
    def oru_update(self, observation, current_timestep):
        """
        - If a track was lost for k frames, don't jump directly from past_observation -> new detection.
        - Instead, generate k "virtual" observations between them and update the KF 
        """

        # If no KF, just initialize + update once.
        if not self.has_filter:
            self.init_filter(observation)
            self.timestep = current_timestep
            self.update(observation)
            return

        # How many frames were we "missing" detections
        # Example: last_obs_timestep=10, current_timestep=14 => missing frames are 11,12,13 => k=3
        k = max(0, current_timestep - self.last_obs_timestep - 1)

        # Use the last real detection bbox as the start
        start = np.array(self.past_observations[-1], dtype=float) # last real detection bbox of fly
        end = np.array(observation, dtype=float) # latest detection bbox of fly

        # virtual observations for the missing frames:
        for i in range(1, k + 1):
            alpha = i / (k + 1) # makes a number between 0 and 1 that increases as i goes from 1 to k, so we get virtual obs that are evenly spaced between start and end
            virt = (1 - alpha) * start + alpha * end # Put a box path between the last real detection and the new detection
            virt_bbox = virt.tolist() # convert back to list format for the update, simulated obs for the missing frames

            # pretending we are at the missing frames i
            self.timestep = self.last_obs_timestep + i

            # predict to next frame + update with virtual measurement, feed the KF these virtual observations 
            self.kalman_filter.update(self._z_from_bbox(virt_bbox))
            self.past_observations.append(virt_bbox)

        # use the last real observation update at current frame
        self.timestep = current_timestep
        self.update(observation)
