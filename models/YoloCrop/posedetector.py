""" Mediapipe model  """

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import math
import numpy as np
from config.config import Config
import types

class PoseDetector:

    def __init__(self)->None:

        self.mpDraw : types.ModuleType = mp.solutions.drawing_utils
        self.mpPose : types.ModuleType  = mp.solutions.pose

        self.utils = Config.Mediapipe.Utils
        self.draw_keypoints = Config.Yolo.Detection.DRAW_BBOX
        detection_config = Config.Mediapipe.Detection

        self.pose : mp.solutions.pose.Pose = self.mpPose.Pose(static_image_mode = detection_config.STATIC_IMAGE_MODE, 
                                                            model_complexity = detection_config.MODEL_COMPLEXITY, 
                                                            smooth_landmarks = detection_config.SMOOTH_LANDMARKS, 
                                                            min_detection_confidence = detection_config.MIN_DETECTION_CONFIDENCE,
                                                            min_tracking_confidence = detection_config.MIN_TRACKING_CONFIDENCE)
                                     

    def get_landmarks_from_image(self, img:np.ndarray)->None:
        """Run pose detection model to detect keypoints from the input frame"""
       
        self.results : landmark_pb2.NormalizedLandmarkList = self.pose.process(img)

        if self.draw_keypoints:
        
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)


    
    def __call__(self, img:np.ndarray)->dict[str, list[int, int]]: 
        """Get landmarks from the image then returns the detection coordinates"""
       
        self.get_landmarks_from_image(img)
        landmarks : dict[str, list[int, int]] = self.get_coordinates_from_landmarks(img) 
        
        return landmarks


    def get_coordinates_from_landmarks(self, image: np.ndarray)->dict[str, list[int, int]]:
        """ Returns the coodinates of each detected hand. The original function is at https://github.com/google/mediapipe/blob/v0.8.6/mediapipe/python/solutions/drawing_utils.py """

        if self.results.pose_landmarks:
            
            if image.shape[2] != self.utils.RGB_CHANNELS:
                raise ValueError('Input image must contain three channel rgb data.')

            image_rows , image_cols, _  = image.shape
            idx_to_coordinates : dict[int, tuple[int, int]] = {}

            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                if ((landmark.HasField('visibility') and
                    landmark.visibility < self.utils.VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                    landmark.presence < self.utils.PRESENCE_THRESHOLD)):
                    continue

            landmark_px : tuple[int, int] = self._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
        
            if self.mpPose.POSE_CONNECTIONS:
        
                result : dict[str, list[int, int]] = {connection[0].name: list(idx_to_coordinates.get(connection[0].value, [-1, -1])) 
                                                     for connection in self.mpPose.POSE_CONNECTIONS 
                                                     if connection[0].name in self.utils.KEYPOINTS_OF_INTEREST}
                return result        
                        
        result : dict[str, list[int, int]] = {}

        for item in self.utils.KEYPOINTS_OF_INTEREST:
            result[item]  = result.get(item, [-1, -1])

        return result
     

    @staticmethod 
    def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int)->tuple[int, int]:
        """Converts normalized value pair to pixel coordinates. More info at https://github.com/google/mediapipe/blob/v0.8.6/mediapipe/python/solutions/drawing_utils.py"""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None

        x_px : int = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px : int = min(math.floor(normalized_y * image_height), image_height - 1)

        return x_px, y_px




