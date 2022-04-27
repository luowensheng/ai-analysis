from re import L
from types import FunctionType
import cv2
from models import ComputeAgeGenderKeypointsMN, ComputeAgeGenderKeypointsMP
from monads import Option
import numpy as np
from urllib.request import urlopen

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


def url_to_image(url, readFlag=cv2.IMREAD_COLOR):

    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    return image


def load_image_from_path(p):
    if "http" in p[:6]:
        return Option(lambda : url_to_image(p))
    return Option(lambda : cv2.imread(p))

def predict_given_path(p:str, func:FunctionType):
    if "http" in p[:6]:
        return Option(lambda : url_to_image(p)).apply(func)
    return Option(lambda : cv2.imread(p)).apply(func)

def get_movenet_prediction(img):
    return ComputeAgeGenderKeypointsMN(img)

def get_mediapipe_prediction(img):
    return ComputeAgeGenderKeypointsMP(img)
 

def draw_rect(_frame, data, elapsed, prediction, imageLocation):
    frame = _frame.copy()
    x1, y1, x2, y2 = data['face']
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA)
    prediction.write(f"age:{data['age']}, gender:{data['gender']}, fps: {int(1/(elapsed+10**(-10)))}")
    imageLocation.image(frame)        
