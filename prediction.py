from types import FunctionType
from typing import Tuple
import numpy as np
from models import ComputeAgeGenderKeypointsMN, ComputeAgeGenderKeypointsMP, ComputeAgeGenderKeypointsMTCNN, Mean
import streamlit as st
import os
import cv2
from time import time
from utils import load_image_from_path

PREDICTION_FUNCTIONS:list[Tuple[str, FunctionType]] = [
                        ("Mediapipe Face Detection", ComputeAgeGenderKeypointsMP), 
                        ("Movenet Face Detection", ComputeAgeGenderKeypointsMN), 
                        ("MTCNN Face Detection", ComputeAgeGenderKeypointsMTCNN), 
                ]


class Predictor:
 
    def __init__(self, title:str, prediction_function:FunctionType, col, shape:Tuple[int, int]=(500, 500), bbox_color:Tuple[int, int, int]=(0, 0, 255)) -> None:
 
        self.prediction_function=prediction_function
        self.title = title
        self.col = col
        self.col.write(self.title)
        self.prediction = self.col.empty()
        self.imageLocation = self.col.empty()
        self.shape = shape
        self.bbox_color= bbox_color


    def predict(self, image: np.ndarray)->dict:
        return  self.prediction_function(image)    

    def predict_and_draw(self, image: np.ndarray)-> float:

        t1 = time()
        data = self.predict(image)
        elapsed = time() - t1
        self.draw_bbox(image, data, elapsed)  

        return elapsed

    def draw_bbox(self, image:np.ndarray, data:dict, elapsed:float)->None:

        frame = image.copy()
        x1, y1, x2, y2 = data['face']
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bbox_color, 2) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.shape, interpolation=cv2.INTER_AREA)
        self.imageLocation.image(frame)        
        self.prediction.markdown(f"""<p>age:<b>{data['age']}</b>, gender:<b>{data['gender']}</b>, fps:<b>{int(1/(elapsed+10**(-10)))}</b></p>""", unsafe_allow_html=True)



def setup_predictors(prediction_functions:list[Tuple[str, FunctionType]]):
 
    n = len(prediction_functions)
    predictors: list[Predictor] = []

    for i, col in enumerate(st.columns(n)):
        
        item = prediction_functions[i]
        title = item[0]
        func = item[1]
        predictors.append(Predictor(title, func, col))
    return predictors    



def setup_prediction(prediction_functions:list[Tuple[str, FunctionType]]=PREDICTION_FUNCTIONS):
    
    actual = st.empty()
    predictors = setup_predictors(prediction_functions)

    def predict(p):
        
        try:
            age, gender, *_ = os.path.split(p)[-1].split("_")
            age = int(age)
            gender = 'male' if gender=='0' else 'female'
            actual.markdown(f"""
            <h2 style="text-align:center;">
            Actual age:<b>{age}<b>, Actual gender:<b>{gender}</b>
            </h2>""", unsafe_allow_html=True)

        except Exception as e: 
            #  actual.mardow("<p>No label provided for this image</p>", unsafe_allow_html=True)
            pass
        
        try: 
            frame_option = load_image_from_path(p)
            frame = frame_option.unwrap()
            if frame is None:
                actual.write(f"An error has occured, please try again with another image. ERROR: {frame_option.logs}")
                return    

            for predictor in predictors:
                predictor.predict_and_draw(frame)
            
        except Exception as e: 
            st.write(e)
            
    return predict



def setup_video(prediction_functions:list[Tuple[str, FunctionType]]= PREDICTION_FUNCTIONS):
    
    predictors = setup_predictors(prediction_functions)

    mean_time  = Mean()

    def predict(video):

        for predictor in  predictors:

            cap = cv2.VideoCapture(video)

            mean_time.reset()

            while True:              
                try: 
                    ret, frame = cap.read()
                    if frame is None or not ret:
                        cap.release()
                        break    

                    predictor.predict_and_draw(frame) 
    
                except Exception as e: 
                    st.write(e)
                
    return predict

