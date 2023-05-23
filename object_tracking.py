import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")

# Initialize count
count = 0
track_id = 0

object_tracking = []


while True:
    center_points_cur_frame = []

    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x,y,w,h) = box
        cen_pt_x = int((x+x+w) / 2)
        cen_pt_y = int((y+y+h) / 2)
        center_points_cur_frame.append((cen_pt_x, cen_pt_y))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

        if count < 2:
            temp_list = []
            temp_list.append((cen_pt_x, cen_pt_y))
            #print(temp_list)
            temp_track = [track_id, temp_list]
            object_tracking.append(temp_track)
            track_id = track_id + 1

    object_tracking_copy = object_tracking.copy()
    center_points_cur_frame_copy = center_points_cur_frame.copy()
    index = 0
    if count >= 2:
        for obj_track, obj in object_tracking_copy:
            #print(obj_track)
            #print(obj)
            recent_pt = len(obj) - 1
            obj_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt[0]-obj[recent_pt][0], pt[1]-obj[recent_pt][1])

                if distance < (20+pt[1]/50):
                    obj_exists = True
                    if recent_pt == 2:
                        temp_list = object_tracking[index][1]
                        object_tracking[index][1].append(pt)
                        #print(temp_list)
                        object_tracking[index][1] = temp_list[1:3:1]
                    object_tracking[index][1].append(pt)
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    break

            if not obj_exists:
                temp_obj = [obj_track, obj]
                object_tracking.remove(temp_obj)
            else:
                index = index + 1

        for pt in center_points_cur_frame:
            temp_list = [pt]
            temp_track = [track_id, temp_list]
            object_tracking.append(temp_track)
            track_id = track_id + 1

        #separate object prediction and print obj ids for all
        for obj_track, obj in object_tracking:
            recent_pt = len(obj) - 1
            cv2.circle(frame, obj[recent_pt], 5, (0,0,255),-1)
            cv2.putText(frame, str(obj_track), (obj[recent_pt][0], obj[recent_pt][1]-7), 0, 1, (0,0,255), 2)
            if len(obj) == 3:
                #Velocity average model
                s_x = obj[2][0] - obj[1][0]
                s_y = obj[2][1] - obj[1][1]
                p_x = obj[1][0] - obj[0][0]
                p_y = obj[1][1] - obj[0][1]
                n_x = int((s_x + p_x) / 2)
                n_y = int((s_y + p_y) / 2)
                #Collect greater number of velocities to average? Display line out further?

                #ACCELERATION BASED MODEL
                #s_x = obj[2][0] - obj[1][0]
                #s_y = obj[2][1] - obj[1][1]
                #c_x = s_x - (obj[1][0] - obj[0][0])
                #c_y = s_y - (obj[1][1] - obj[0][1])
                #n_x = s_x + c_x
                #n_y = s_y + c_y 


                cv2.circle(frame, (obj[recent_pt][0]+n_x, obj[recent_pt][1]+n_y ), 5, (255,0,0),-1)
                cv2.putText(frame, (str(obj_track) + "'"), (obj[recent_pt][0]+n_x, obj[recent_pt][1]+n_y - 7), 0, 1, (255,0,0), 2)

    cv2.imshow("Frame", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()





