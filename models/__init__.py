from .AgeGenderDetector import predict_age_gender
from .Mediapipe import get_face_location
from .Movenet import predict_keypoints, draw_keypoints
import cv2
from .tracking import tracker
import numpy as np

class Mean:
    def __init__(self, resolve=lambda x:x) -> None:
        self.val = 0
        self.k = 0
        self.resolve = resolve

    def update(self, val, reset=False):
        self.k += 1
        if (self.k == 1) or reset:
            self.val = val
        else:    
           self.val = self.val+ (1/self.k)*(val-self.val)
        return self.resolve(self.val)


def ComputeAgeGenderKeypoints(frame):
    
    try:
        x1, y1, x2, y2 = get_face_location(frame)
        data = predict_age_gender(frame[y1:y2, x1:x2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 

    except Exception as e:
        print(e)
        data = predict_age_gender(frame)

    
    return { 
            "frame": frame, 
             "age": data['age'], 
             "gender": 'male' if data['gender'] ==0 else 'female', 
            }


def ComputeAgeGenderKeypointsMP(frame):
    try:
        x1, y1, x2, y2 = get_face_location(frame)
        data = predict_age_gender(frame[y1:y2, x1:x2])

    except Exception as e:
        print(e)
        data = predict_age_gender(frame)
        x1, y1, x2, y2 = [0, 0, frame.shape[0], frame.shape[1]]
    
    return { 
             "age": data['age'], 
             "gender": 'male' if data['gender'] ==0 else 'female', 
             "face": [ x1, y1, x2, y2] ,
             "shape": frame.shape
            }


# def ComputeAgeGenderKeypointsMN(frame):

#     detections, frame = predict_keypoints(frame)
    
#     target = tracker.ComputeTargetLocation(detections, reset=1)
    
#     facebbox = tracker.get_target_face()

#     x1, y1, x2, y2 = facebbox

#     data = predict_age_gender(frame[y1:y2, x1:x2])

#     return { 
#             "frame": frame, 
#              "age": data['age'], 
#              "gender": 'male' if data['gender'] ==0 else 'female', 
#             }

    
def ComputeAgeGenderKeypointsMN(frame):
    
    try:
        detections = predict_keypoints(frame)
        
        tracker.ComputeTargetLocation(detections, reset=True)
        
        facebbox = tracker.get_target_face()

        x1, y1, x2, y2 = facebbox
        shape = frame.shape
        x1, y1, x2, y2 = np.array([x1 * shape[0], y1 * shape[1], x2 * shape[0], y2 * shape[1]], dtype=int)
    except:
        x1, y1, x2, y2 = [0, 0, frame.shape[0], frame.shape[1]]

    data = predict_age_gender(frame[y1:y2, x1:x2])
        

    
    return { 
            #"keypoints": target,
             "age": data['age'], 
             "gender": 'male' if data['gender'] ==0 else 'female', 
             "face": [ int(x1), int(y1), int(x2), int(y2)] 
            }

    

