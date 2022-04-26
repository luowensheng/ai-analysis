from types import FunctionType
import requests
from subprocess import check_output
import json
import cv2
from models import ComputeAgeGenderKeypointsMN, ComputeAgeGenderKeypointsMP
from monads import Option

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


def predict_given_path(p, func:FunctionType):
    return Option(lambda : cv2.imread(p)).perform(func)

def get_movenet_prediction(p):
    return predict_given_path(p, ComputeAgeGenderKeypointsMN) 

def get_mediapipe_prediction(p):
    return predict_given_path(p, ComputeAgeGenderKeypointsMP)
 

def draw_rect(_frame, data, elapsed, prediction, imageLocation):
    frame = _frame.copy()
    x1, y1, x2, y2 = data['face']
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA)
    prediction.write(f"age:{data['age']}, gender:{data['gender']}, fps: {int(1/(elapsed+10**(-10)))}")
    imageLocation.image(frame)        