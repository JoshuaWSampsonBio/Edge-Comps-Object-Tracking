def DetectionReader(path):
    video_observations = [] #list of frame_observations
    # This becomes the full video: list of frames, each frame is a list of detections.

    frame_observations = [] #list of detections in a single frame
    # This is your running list for the current frame.

    counter = 1
    # This tracks which frame we’re currently accumulating.
    # Assumes the first frame in the file is frame 1.

    with open(path) as f:
        lines = f.readlines()
    # Read all lines from file into memory.

    last_line = lines[-1]
    last_detection = last_line.split(',')
    last_frame = int(last_detection[0])

    for detection in lines:
        m = detection.split(',')
        # m[0] = frame
        # m[1..5] = x1,y1,x2,y2,score

        if (int(m[0]) > counter):
            # This means the file moved to a new frame.
            # So we "close out" the previous frame’s detections.

            counter += 1
            video_observations.append(frame_observations)
            # store completed frame list

            frame_observations = []
            # reset for next frame

            while (counter < int(m[0])):
                # If frames are skipped (e.g., file jumps from frame 5 to frame 8),
                # you insert empty lists for missing frames.
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
