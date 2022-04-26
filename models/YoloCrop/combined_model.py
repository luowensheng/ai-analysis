""" Combined Yolo and mediapipe Model """
from .posedetector import PoseDetector
from config.config import Config
import numpy as np

import torch
import cv2
import threading

import pandas  


class CombinedModel:

    def __init__(self) -> None:

        self.yolo_model = torch.hub.load(Config.Yolo.Model.PATH, 
                                         Config.Yolo.Model.SIZE, 
                                         Config.Yolo.Model.PRETRAINED)

        self.detection_config  = Config.Yolo.Detection

        if Config.Yolo.Model.GPU:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = 'cpu'   

        self.yolo_model.to(device);
        
        self.mediapipe_model : PoseDetector = PoseDetector()



    def __call__(self, img: np.ndarray)->tuple[list[dict[str, list[int, int]]], np.ndarray]:
        """Retrieves the bounding box locations from the yolo model then the mediape model return the location of the detected hands"""
       
        self.img : np.ndarray = img
        # get yolo prediction
        yolo_detections : pandas.DataFrame = self.yolo_model(self.img).pandas().xyxy[0][:self.detection_config.MAX_DETECTIONS]

        self.bbox_points_list : list[pandas.DataFrame] = []   # used to store the bounding boxes generated from the yolo predictions  
        self.keypoints_list : list[dict[str, list[int, int]]] = []     # used to store the keypoints generated from the mediapipe predictions  
        
        threads : list[threading.Thread] = []
        # the lock is used to ensure that multiple threads do not try to access the image at the same time
        self.lock : threading.Lock = threading.Lock()

        n_detections =  len(yolo_detections)
        for i in range(n_detections):

            yolo_detection : pandas.DataFrame = yolo_detections.iloc[i]
            
            if not (yolo_detection["confidence"]>self.detection_config.BOX_MIN_CONFIDENCE and
                                  yolo_detection["class"]==self.detection_config.PERSON_CLASS):
                    continue
            
            thread : threading.Thread = threading.Thread(target=self.get_keypoints_from_img_patch, args=(img, yolo_detection))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if self.detection_config.DRAW_BBOX:
               for points in self.bbox_points_list:

                   cv2.rectangle(   img=self.img, 
                                    pt1=(points["xmin"],points["ymin"]),
                                    pt2=(points["xmax"], points["ymax"]), 
                                    color=self.detection_config.BOX_COLOR, 
                                    thickness=self.detection_config.BOX_THICKNESS
                                )

        return self.keypoints_list   


 
    def get_keypoints_from_img_patch(self, img, yolo_detection:pandas.DataFrame)->None:
        """A patch of containing a yolo detection is sent to the mediapipe model for hand localization """
        
        #img : np.ndarray = cv2.resize(img, self.detection_config.PATCH_SHAPE , cv2.INTER_AREA)
        points : pandas.DataFrame = yolo_detection[["xmin","ymin","xmax", "ymax"]].astype(int)
        
        if self.detection_config.DRAW_BBOX:
            self.bbox_points_list.append(points)
        
        # An offset is added to bounding box points to get a slighly bigger patch to ensure the the target is inside the patch

        y_start : int = max(points["ymin"]-self.detection_config.BBOX_CROP_OFFSET, 0)
        y_end : int = points["ymax"]+self.detection_config.BBOX_CROP_OFFSET
        x_start : int = max(points["xmin"]-self.detection_config.BBOX_CROP_OFFSET, 0) 
        x_end : int = points["xmax"]+self.detection_config.BBOX_CROP_OFFSET


        # each thread has to acquire the lock before making a prdiction
        

        # a segment of the whole image containing a detected person is taken from the original image to be passed to the mediapipe model        
        img_patch : np.ndarray = img[y_start:y_end, x_start:x_end, :]
        self.lock.acquire()
        prediction : dict[str, list[int, int]] = self.mediapipe_model(img_patch)
        self.lock.release()
        
        # Each detection are rescaled to their appropriate location in the original image 
        for item in prediction:
            if prediction[item][0]!=-1:  # -1 represents no detection
                
                prediction[item][1]+= points["ymin"] 
                prediction[item][0]+= points["xmin"] 
            
        self.keypoints_list.append(prediction) 