from typing import Tuple

from numpy import ndarray
from .movenet_utils import tf_to_numpy, draw_keypoints, prepocess, load_model, get_keypoints_and_detections
import cv2

movenet = load_model()

def predict(image_src, shape=None)->Tuple[ list[dict], ndarray]:
    
    image = prepocess(image_src)

    # Run model inference.
    keypoints = get_keypoints(movenet, image)
    img = tf_to_numpy(image)
   
    detections : list[dict] = get_keypoints_and_detections(keypoints, shape or img.shape, None)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return detections #, img


def get_keypoints(model, image):
    try:
     outputs = model(image)
    except Exception as e:
        print(e)
        return 
    # Output is a [1, 6, 56] tensor.
    keypoints = outputs['output_0']
    return keypoints

