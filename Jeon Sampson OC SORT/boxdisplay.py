import cv2 as cv
from colorhash import ColorHash
import os  # CHANGED: needed to save outputs


# Assume detections are of format: [[x1,y1,x2,y2], frame, track_id]
# - detection[0] is the bounding box corners (xyxy)
# - detection[1] is the frame number
# - detection[2] is the track's unique ID (identity)


def GetIndex(detection):
    return detection[1]
    # This is a helper function used for sorting.
    # Sorting by frame ensures you process detections in chronological frame order.
    # Example: all frame 1 boxes, then all frame 2 boxes, etc.


def BoxDisplay(path, detections, out_dir="tracked_frames"):  # CHANGED: output folder
    # path is the prefix of frame images, like:
    # "/home/.../006/006_" so frame 1 is ".../006_001.jpg"
    #
    # detections is the full list of [bbox, frame, id] items from OC-SORT
    #
    # out_dir is where you will save the annotated frames.

    os.makedirs(out_dir, exist_ok=True)  # CHANGED
    # Make sure output directory exists.
    # exist_ok=True means "if it's already there, don't crash".

    detections.sort(key=GetIndex)
    # This orders the detection list by frame number so we can process sequentially.

    counter = -1
    # counter stores the "current frame number we have loaded".
    # It starts at -1 so the first detection’s frame will be different and trigger loading.

    img = None
    # img will hold the currently loaded image (a numpy array).
    # If img is None, no image is loaded yet.

    img_path = None  # CHANGED
    # img_path stores the filename of the currently loaded image,
    # so we can save it later with the same name into out_dir.


    for detection in detections:
        # Loop through every detection/track output.
        # Each item: [[x1,y1,x2,y2], frame, track_id]

        ###DEBUGGING
        ###if detection[2] != 1:
           ### continue
            ###

        frame = detection[1]
        # Extract the frame index for this detection.
        # We’ll use this to decide which image to load.

        # CHANGED: load a new image when frame changes
        if frame != counter:
            # If the detection belongs to a NEW frame,
            # we need to:
            # 1) save the old frame (if we have one)
            # 2) load the new frame image

            # CHANGED: save previous frame before loading next
            if img is not None and img_path is not None:
                # If we have an image currently loaded, save it now
                # BEFORE switching to the next frame.

                out_path = os.path.join(out_dir, os.path.basename(img_path))
                # Example:
                # img_path = "/home/.../006_007.jpg"
                # basename = "006_007.jpg"
                # out_path = "tracked_frames/006_007.jpg"

                cv.imwrite(out_path, img)
                # Write the annotated image to disk.

            counter = frame
            # Update current frame counter.

            img_path = path + f'{counter:03d}' + '.jpg'
            # Build the input image filename.
            # counter:03d means pad with zeros to 3 digits:
            # 1 -> "001", 12 -> "012", 125 -> "125"
            #
            # Example:
            # path="/home/.../006/006_"
            # counter=7 -> "/home/.../006/006_007.jpg"

            img = cv.imread(img_path)  # CHANGED: actually load the image
            # Load the image pixels into a numpy array.
            # If the path is wrong or file is missing, img will become None.

            if img is None:
                print("FAILED TO LOAD:", img_path)
                # Warn you that image wasn't found.
                # This often happens if you have fewer frames than your detections reference
                # or the naming format doesn't match.
                continue
                # Skip drawing for this detection since we can’t draw without an image.

        # CHANGED: convert bbox to ints (OpenCV expects ints)
        x1, y1, x2, y2 = map(int, detection[0])
        # OpenCV drawing functions expect integer pixel coordinates.
        # Your tracker uses floats, so this safely converts them to ints.

        # CHANGED: ColorHash -> tuple of ints for OpenCV
        color = tuple(int(c) for c in ColorHash(str(detection[2])).rgb)
        # detection[2] is the track_id.
        # We convert it to a string and hash it into a stable RGB color.
        #
        # This is huge for tracking visualization:
        # - Track 3 will always be the same color across frames.

        cv.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # Draw rectangle on the current frame image.
        # thickness=3 means 3 pixels thick.

    # CHANGED: save the last frame
    if img is not None and img_path is not None:
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv.imwrite(out_path, img)
        # After the loop ends, the last loaded frame wasn’t saved yet
        # (because saving happens when you switch frames).
        # So we save it here.
