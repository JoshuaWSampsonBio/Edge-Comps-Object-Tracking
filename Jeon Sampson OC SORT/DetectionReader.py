def DetectionReader(path):
    video_observations = [] #list of frame_observations
    # full video: list of frames, each frame is a list of detections.

    frame_observations = [] #list of detections in a single frame
    # running list for the current frame.

    counter = 1

    with open(path) as f:
        lines = f.readlines()
    # Read all lines from file into memory.

    last_line = lines[-1]
    last_detection = last_line.split(',')
    last_frame = int(last_detection[0])

    for detection in lines:
        m = detection.split(',')

        if (int(m[0]) > counter):

            counter += 1
            video_observations.append(frame_observations)
            # store completed frame list

            frame_observations = []
            # reset for next frame

            while (counter < int(m[0])):
                video_observations.append([])
                counter += 1

        detection_vector = []
        for i in range(1, 6):
            detection_vector.append(float(m[i]))
            # build [x1,y1,x2,y2,score]

        frame_observations.append(detection_vector)
        # add this detection to current frame list.

    video_observations.append(frame_observations)
    # add the last frame when loop ends

    return video_observations
    # return list-of-frames format
