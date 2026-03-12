def DetectionReader(path):
    video_observations = [] #list of frame_observations

    frame_observations = [] #list of detections in a single frame

    counter = 1

    with open(path) as f:
        lines = f.readlines()

    last_line = lines[-1]
    last_detection = last_line.split(',')
    last_frame = int(last_detection[0])

    for detection in lines:
        m = detection.split(',')

        if (int(m[0]) > counter):

            counter += 1
            video_observations.append(frame_observations)

            frame_observations = []

            while (counter < int(m[0])):
                video_observations.append([])
                counter += 1

        detection_vector = []
        for i in range(1, 6):
            detection_vector.append(float(m[i]))

        frame_observations.append(detection_vector)

    video_observations.append(frame_observations)
    # add the last frame when loop ends

    return video_observations
