import time
from typing import Tuple
from .config import KEYPOINT_NAMES, CONNECTIONS
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

RESIZE_SHAPE: tuple[int, int] = (256, 256) # (256, 256)


def load_model(model_path="https://tfhub.dev/google/movenet/multipose/lightning/1"):
    """[summary]

    Args:
        model_path (str, optional): [description]. Defaults to "https://tfhub.dev/google/movenet/multipose/lightning/1".

    Returns:
        [type]: [description]
    """
    print(f"loading model from {model_path}...")
    model = hub.load(model_path)
    movenet = model.signatures["serving_default"]
    print(f"Finished.")

    return movenet


def prepocess(image_src, resize_shape=RESIZE_SHAPE):
    """[summary]

    Args:
        image_src ([type]): [description]

    Returns:
        [type]: [description]
    """
    image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    if isinstance(image_src, str):
        image = tf.io.read_file(image_src)
        image = tf.compat.v1.image.decode_jpeg(image)
    elif isinstance(image_src, np.ndarray):
        image = tf.convert_to_tensor(image_src)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, *resize_shape), dtype=tf.int32)
    
    
    return image


def tf_to_numpy(t):
    """[summary]

    Args:
        t ([type]): [description]

    Returns:
        [type]: [description]
    """
    return cv2.cvtColor(t[0].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)


def rescale_keypoints(
    person_keypoints, shape, requested_body_parts=None
) -> Tuple[dict[str, list[int, int, float]], float]:
    """[summary]

    Args:
        person_keypoints ([type]): [description]
        shape ([type]): [description]

    Returns:
        Tuple[dict[str, list[int, int, float]], float ]: [description]
    """
    person_keypoints.resize(17, 3)
    data = {}
    t_score = 0
    for i, point in enumerate(person_keypoints):
        body_part = KEYPOINT_NAMES[i]
        if  requested_body_parts:
           if not body_part in requested_body_parts:
               continue 
        y, x, score = point
        t_score += score
        data[body_part] = [x, y, score]
        # data[body_part] = [int(x * shape[0]), int(y * shape[1]), score]
    return data, t_score


def rescale_bbox(bbox, shape, format="xyxy"):
    """
    ymin, xmin, ymax, xmax, score
    """
    if format == "xyxy":
        y1, x1, y2, x2 = bbox

        bbox = np.array([x1 , y1, x2, y2], dtype=float)
        # bbox = np.array([x1 * shape[0], y1 * shape[1], x2 * shape[0], y2 * shape[1]], dtype=int)
        return bbox
    raise NotImplementedError(f"not implement for format = [{format}], try [x_min, x_max, y_min, y_max]")

def collect_bbox_and_keypoints(keypoints, ps, shape, requested_body_parts=None, return_t_score=False):
    """[summary]

    Args:
        keypoints ([type]): [description]
        ps ([type]): [description]
        shape ([type]): [description]
        requested_body_parts ([type], optional): [description]. Defaults to None.
        return_t_score (bool, optional): [description]. Defaults to False.

    Yields:
        [type]: [description]
    """    
    for i in range(keypoints.shape[1]):
        person_keypoints, t_score = rescale_keypoints(keypoints[0][i][:ps].numpy(), shape, requested_body_parts)
        person_bbox: np.array = rescale_bbox(keypoints[0][i][ps:-1].numpy(), shape)
        person_score: float = keypoints[0][i][1].numpy()
        detection = {"keypoints": person_keypoints, "bbox": person_bbox, "score": person_score}

        if return_t_score:
           yield detection, t_score
        else:   
            yield detection




    # def load_model(model_path):



def get_keypoints_and_detections(keypoints, shape,  requested_body_parts, score_threshold=0.1):
    detections = []
    ps = 17 * 3

    for (detection, t_score)  in collect_bbox_and_keypoints(keypoints, ps, shape, requested_body_parts,return_t_score=True):
       detection['t_score'] = t_score
       if detection['score'] >= score_threshold : 
           detections.append(detection)
    
    return detections




def draw_keypoints(img:np.ndarray, detections:list[dict], requested_body_parts=None, score_threshold=0.1):
 
    if requested_body_parts:
        connections = filter(lambda conn: any([bp for bp in conn if bp in requested_body_parts]), CONNECTIONS)
    else:
        connections  = CONNECTIONS 
    
    for detection in detections:

        draw_box = False
        for connection in connections:

                draw_line = False

                bp1 = detection["keypoints"].get(connection[0], None)
                bp2 = detection["keypoints"].get(connection[1], None)
                if bp1:

                    if bp1[2] > score_threshold:

                            color = (255, 0, 24)
                            cv2.circle(img, (bp1[0], bp1[1]), 5, color, 2)
                            draw_line = True
                            draw_box = True

                            
                if bp2:
                    if bp2[2] > score_threshold:

                            color = (255, 0, 24) 
                            cv2.circle(img, (bp2[0], bp2[1]), 5, color, 2)
                            draw_line = True and draw_line
                            draw_box = True

                
                if  draw_line:
                    cv2.line(img, (bp1[0], bp1[1]), (bp2[0], bp2[1]), (255, 0, 0), 2)
        if draw_box:
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 

    return img     