import time
from cv2 import INTER_AREA
from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


detector = MTCNN()


def process_bbox(bbox):
    x,y,w,h  = bbox
    return [x,y, x+w, y+h]

def get_single_bbox(img):
    results = detector.detect_faces(img)
    return process_bbox(results[0]['box'])

def get_all_bbox(img):
    results = detector.detect_faces(img)    
    return [process_bbox(res['box']) for res in results]

def get_full_predictions(img):
    return detector.detect_faces(img)   






###  OUTPUT
# [
#     {
#         'box': [277, 90, 48, 63],
#         'keypoints':
#         {
#             'nose': (303, 131),
#             'mouth_right': (313, 141),
#             'right_eye': (314, 114),
#             'left_eye': (291, 117),
#             'mouth_left': (296, 143)
#         },
#         'confidence': 0.99851983785629272
#     }
# ]