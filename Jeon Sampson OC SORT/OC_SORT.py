import numpy as np
import IOU
import DeltaTheta
from Track import Track
from scipy.optimize import linear_sum_assignment
import DetectionReader
import boxdisplay

detections = DetectionReader.DetectionReader("/home/sampsonj2/Desktop/detections_006.txt")

###DEBUGGING
#tracks_made = 0
#detection_count = 0
#obs_match_count = 0
#min_cost = 0 

#for i in detections:
#    detection_count += len(i)
#print("num detections:"+str(detection_count))
#all_matched_det_id = set()  


###

# making sure the detections are [x1, y1, x2, y2, score] format
def strip_det(det):
    det = np.asarray(det, dtype=float).reshape(-1)
    if det.size >= 6:
        return det[1:6]
    return det

# only taking the bbox part of the detection for matching and tracking
def det_bbox(det):
    return strip_det(det)[:4].tolist()


# T_matched_now = list of tracks that have been matched to a detection in this frame
# matched_obs = list of detections that have been matched to a track in this frame
# T_unmatched_t = list of tracks that were NOT matched to any detection in this frame
# Z_unmatched_t = list of detections that were NOT matched to any track in this frame
def step3_step4_update_manage(T_matched_now, matched_obs, T_unmatched_t, Z_unmatched_t, texpire, t):
    #global tracks_made
    for tau, det_raw in zip(T_matched_now, matched_obs):
    # zip method pairs things up like: (track0, detection0), (track1, detection1)
	# each track + a detectioin in the current frame for it
        obs_bbox = det_bbox(det_raw) if len(np.asarray(det_raw).reshape(-1)) >= 5 else det_raw
			# just making sure it's properly taking the bbox of the matched detection
            

        # was_lost = previously unmatched, needs the ORU update
        # track may have found a detection in this frame, but was unmatched in the previous frame(s), 
		# so we want to do the ORU update to "rescue" it and get its filter back on track.
        was_lost = (tau.tracked == False) or (tau.untracked > 0)

        tau.tracked = True
        tau.untracked = 0
        tau.timestep = t

        # ORU happens if the track was previously unmatched (lost) and now has a match again, 
		# so we can use the detection to "rescue" the track and update its filter.
        # else just do a normal update with the new detection.
        if was_lost:
            tau.oru_update(obs_bbox, t)
        else:
            if tau.check_filter():
                tau.update(obs_bbox)
            else: # might not have a KF yet although this should be rare since we init KF at track creation, but just in case
                tau.init_filter(obs_bbox)
                tau.update(obs_bbox)
                

    # STEP 4: init new tracks from unmatched detections (this is for when a new fly enters the frame and we want to start tracking it)
    T_new_t = [] # all the new tracks created in this frame
    for det_raw in Z_unmatched_t:
        obs_bbox = det_bbox(det_raw) if len(np.asarray(det_raw).reshape(-1)) >= 5 else det_raw
        tr = Track(obs_bbox, t) # a new track with this detection as its first observation
        #tracks_made += 1 ###DEBUGGING
        tr.init_filter(obs_bbox) #kf
        T_new_t.append(tr) 


    # STEP 4: expire unmatched tracks
    T_reserved_t = []
    T_expired_t = []
    for tau in T_unmatched_t:
        tau.tracked = False
        tau.untracked += 1
        tau.timestep = t
        if tau.untracked < texpire:
            T_reserved_t.append(tau)
        else:
            T_expired_t.append(tau)

	# return the updated list of active tracks 
	# and the expired tracks to log them separately
    return T_new_t + T_matched_now + T_reserved_t, T_expired_t



# josh's code

# params
texpire = 30
consistency_lambda = 0.2
IOU_threshold = 1000000
delta_t = 3
intention_distance = 2

t = 0
Expired = []
T = []  # !!!!!!!!!!!!!!!!!!!! ALL active tracks not just "T_matched_t"

Output = [] # !!!!!!!!!!!!!!!!!!!! for the box display later

for observations in detections:
    t += 1

    # !!!! make sure just bbox
    observations = [det_bbox(o) for o in observations]

    # !!!!!!!!! init first time we see detections. T_matched_t starts empty
    if len(T) == 0 and len(observations) > 0: # the frame has detections just no active tracks!
        for det in observations:
            tr = Track(det, t)
            tr.init_filter(det)
            T.append(tr)
            #tracks_made += 1 ###DEBUGGING
        for tr in T:
            Output.append([tr.get_past(1), t, tr.id])
        continue

    # !!!!!!!! predict once per frame
    # predict before detection association so that we can 
	# use the predicted positions for matching
    for tr in T:
        tr.timestep = t
        tr.predict()

    tracks = T

    T_matched_now = []
    matched_obs = []
    T_unmatched_t = []
    Z_unmatched_t = []

    if len(tracks) > 0 and len(observations) > 0:
        Cost_matrix = []
        # rows = tracks, cols = detections
        for i in tracks:
            Cost_line = []
            for j in observations:
                iou_val = IOU.IOU(i.get_past(1), j)
                dtheta_val = 0
                if (len(i.get_past_list()) >= delta_t) & (len(i.get_past_list()) >= intention_distance):
                    dtheta_val = DeltaTheta.DeltaTheta(i.get_past(1),i.get_past(delta_t), i.get_past(intention_distance),j) ###fix!!!
                Cost = -1 * (iou_val + consistency_lambda * dtheta_val)
                Cost_line.append(Cost)
            Cost_matrix.append(Cost_line)

        row_ind, col_ind = linear_sum_assignment(Cost_matrix)

        matched_track_idx = set()  
        matched_det_idx = set()  


        if len(tracks) < len(observations):
            for i in range(len(col_ind)):
                c = col_ind[i]
                if IOU.IOU(tracks[i].get_past(1), observations[c]) < IOU_threshold: #Changed from > IOU_threshold to < IOU_threshold
                    T_matched_now.append(tracks[i]) 
                    matched_obs.append(observations[c])
                    matched_track_idx.add(i) 
                    matched_det_idx.add(c) 
                    #all_matched_det_id.add(c) #DEBUGGING
        else:
            for i in range(len(row_ind)):
                r = row_ind[i]
                c = col_ind[i]
                if IOU.IOU(tracks[r].get_past(1), observations[c]) < IOU_threshold: #Changed from > IOU_threshold to < IOU_threshold
                    T_matched_now.append(tracks[r])
                    matched_obs.append(observations[c])
                    matched_track_idx.add(r)
                    matched_det_idx.add(c) 
                    #all_matched_det_id.add(c) #DEBUGGING

                    

        # build a seperate unmatched lists (bug issues)
        for idx, tr in enumerate(tracks):
            if idx not in matched_track_idx:
                T_unmatched_t.append(tr)
        for idx, ob in enumerate(observations):
            if idx not in matched_det_idx:
                Z_unmatched_t.append(ob)
            #else: ###DEBUGGING
            #    obs_match_count += 1


# !!!!!!!!! handling STILL unmatched tracks
# Using 0 gives bugs, so uses matching index sets instead.
# seperate lists to keep track of 
# unmatched tracks and detections instead of setting them to 0 in 
# the original lists.

        if len(T_unmatched_t) > 0 and len(Z_unmatched_t) > 0:
            Cost_matrix2 = []
            for i in T_unmatched_t:
                Cost_line = []
                for j in Z_unmatched_t:
                    Cost_line.append(-1 * IOU.IOU(i.get_past(1), j))
                Cost_matrix2.append(Cost_line)  

			# has tracks, has detections, but no matches
            if len(Cost_matrix2) > 0 and len(Cost_matrix2[0]) > 0:
                r2, c2 = linear_sum_assignment(Cost_matrix2)

                matched_tr2 = set() # the track indices that got matched
                matched_z2 = set() # the detection indices that got matched

                if len(T_unmatched_t) < len(Z_unmatched_t):
                    for i in range(len(c2)):
                        cc = c2[i]
                        if IOU.IOU(T_unmatched_t[i].get_past(1), Z_unmatched_t[cc]) < IOU_threshold: #Changed from > IOU_threshold to < IOU_threshold
                            T_matched_now.append(T_unmatched_t[i])
                            matched_obs.append(Z_unmatched_t[cc])
                            matched_tr2.add(i)
                            matched_z2.add(cc)
                else:
                    for i in range(len(r2)):
                        rr = r2[i]
                        cc = c2[i]
                        if IOU.IOU(T_unmatched_t[rr].get_past(1), Z_unmatched_t[cc]) < IOU_threshold: #Changed from > IOU_threshold to < IOU_threshold
                            T_matched_now.append(T_unmatched_t[rr])
                            matched_obs.append(Z_unmatched_t[cc])
                            matched_tr2.add(rr)
                            matched_z2.add(cc)

                # !!!!!! rebuild remaining unmatched after second pass
                T_unmatched_t = [tr for k, tr in enumerate(T_unmatched_t) if k not in matched_tr2]
                Z_unmatched_t = [ob for k, ob in enumerate(Z_unmatched_t) if k not in matched_z2]

    else:
        T_unmatched_t = tracks[:]
        Z_unmatched_t = observations[:]

    # Step 3/4
    T, T_expired_t = step3_step4_update_manage(T_matched_now, matched_obs, T_unmatched_t, Z_unmatched_t, texpire, t)
    Expired.append(T_expired_t)

    for tr in T:
        if (tr.get_prediction() != None):
            Output.append([tr.get_prediction(), t, tr.id])
        else:
            Output.append([tr.get_past(1), t, tr.id])

###DEBUGGING
#for item in Output:
#    print(item)

#print("Tracks made: "+str(tracks_made))
#print("observations matched: "+str(obs_match_count))
#print("matched detections: "+str(all_matched_det_id))
###

# draw boxes
boxdisplay.BoxDisplay("/home/sampsonj2/Desktop/006/006_", Output)
