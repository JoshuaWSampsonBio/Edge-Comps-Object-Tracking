# deltatheta --> checks if the object is moving in a consistent direction with its past movement.

# This function computes the angle difference between the motion from the predicted position 
# to the observed detection, and the "intention" direction from the predicted position to where the track was previously.

# When tracking, OCSORT wants to answer: “Is this detection the same fly as this track?”

import numpy as np

# helper: convert bbox -> center point
def _center(box):
    # box = [x1, y1, x2, y2]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    return cx, cy
    # returns (center_x, center_y)

def DeltaTheta(pastobservationA,pastobservationB,pastobservationC, observation):
    Ax, Ay = _center(pastobservationA)
    Bx, By = _center(pastobservationB)
    Cx, Cy = _center(pastobservationC)
    # centers of past observations (A and B are used to calculate theta_track, C is used to calculate theta_intention)


    ox, oy = _center(observation)
    # center of the detection we’re trying to match


    # using the past motion direction to predict which detection is plausible.
    # checks for intention consistency: is the track moving in a consistent direction with its past movement?
    trackx = Bx - Ax
    tracky = By - Ay

    # calcuating how far and in what direction new observation is from previous observation C
    intentionx = ox - Cx # change in x from C to observed
    intentiony = oy - Cy # change in y from C to observed


    theta_track = np.arctan2(intentiony, intentionx)    # angle of "A->B" direction reference
    theta_intention = np.arctan2(tracky, trackx)    # angle of "C->observation" motion


    return abs(theta_track - theta_intention)
    # return absolute difference in angles, how much the object would need to turn for this detection to be consistent with its past movement direction.
    # - small => consistent direction
    # - large => inconsistent (maybe wrong match)
