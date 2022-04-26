#%%
import math
from typing import Tuple, Union
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5)


def process_points(location, shape):
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin, relative_bounding_box.ymin, shape[1],
      shape[0])
    rect_end_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin + relative_bounding_box.width,
    relative_bounding_box.ymin + relative_bounding_box.height, shape[1],
      shape[0])
    return (rect_start_point[0], rect_start_point[1], rect_end_point[0], rect_end_point[1])  




def draw_rect(image, location, color=(255, 0, 0), thickness=2):
    image_rows, image_cols, _ = image.shape
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
      image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin + relative_bounding_box.width,
    relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
      image_rows)
    cv2.rectangle(image, rect_start_point, rect_end_point, color, thickness)



def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

# For webcam input:

def get_face_location(image):
  # with mp_face_detection.FaceDetection(
  #     model_selection=0, min_detection_confidence=0.5) as face_detection:
     # image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_AREA)
      shape = image.shape

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          try:
            xmin, ymin, xmax, ymax = process_points(detection.location_data, shape)
            # yk  = (ymax-ymin)//4
            # xk  = (xmax-xmin)//8
            # xmin = max(0,xmin-xk*2)
            # ymin = max(0,ymin-yk)
            # xmax = min(shape[0],xmax+xk)
            # ymax = min(shape[1],ymax+yk)
            return (xmin, ymin, xmax, ymax)
            #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2) 
          except:
            continue

# %%
