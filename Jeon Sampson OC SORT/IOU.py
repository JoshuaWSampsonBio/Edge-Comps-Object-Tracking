# how much does the predicted and previous boxes overlap the detection box?



def Intersection(observation, prediction):
    # observation and prediction are both [x1,y1,x2,y2]
    # prediciton = KF prediction, observation = detection we’re trying to match
    observation_left = observation[0]
    observation_top = observation[1]
    observation_right = observation[2]
    observation_bottom = observation[3]

    prediction_left = prediction[0]
    prediction_top = prediction[1]
    prediction_right = prediction[2]
    prediction_bottom = prediction[3]

    # Overlap rectangle = max(lefts), max(tops), min(rights), min(bottoms)
    intersection_left = max(observation_left, prediction_left)
    intersection_top = max(observation_top, prediction_top)
    intersection_right = min(observation_right, prediction_right)
    intersection_bottom = min(observation_bottom, prediction_bottom)

    width = intersection_right - intersection_left
    height = intersection_bottom - intersection_top
    if (width < 0) | (height < 0):
        return 0
    
    area = width * height
    if area > 0:
        return area
    else:
        return 0
    # if width or height negative => no overlap => intersection area 0


def IOU(observation, prediction):
    """
    observation, prediction are boxes in xyxy format:
      [x1, y1, x2, y2]
    Returns IOU in [0,1].
    """

    obs_w = observation[2] - observation[0]
    obs_h = observation[3] - observation[1]
    pred_w = prediction[2] - prediction[0]
    pred_h = prediction[3] - prediction[1]

    observation_area = max(obs_w, 0) * max(obs_h, 0)
    prediction_area  = max(pred_w, 0) * max(pred_h, 0)
    # max(...,0) prevents negative area if bad box coordinates occur.

    intersection = Intersection(observation, prediction)
    union = observation_area + prediction_area - intersection
    # union = total covered area (sum minus overlap)

#    if intersection>union: #DEBUG
    if union <= 0:
        return 0
    # avoids divide-by-zero if union is degenerate

    return intersection / union


