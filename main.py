from time import sleep, time
import streamlit as st
import os
import cv2
from utils import draw_rect, get_movenet_prediction, get_mediapipe_prediction, load_image_from_path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def setup_prediction():
    
    actual = st.empty()
    col1, col2 = st.columns(2)

    col1.write("Mediapipe Face Detection")
    col2.write("Movenet Face Detection")

    prediction_mp = col1.empty()
    imageLocation_mp = col1.empty()


    prediction_mn = col2.empty()
    imageLocation_mn = col2.empty()


    def predict(p):
        
        try:
            age, gender, *_ = os.path.split(p)[-1].split("_")
            age = int(age)
            gender = 'male' if gender=='0' else 'female'
            actual.markdown(f"""
            <h2 style="text-align:center;">
            Actual age:{age}, Actual gender:{gender}
            </h2>
            """, unsafe_allow_html=True)
        except ValueError: 
            pass
        
        try: 
            frame_option = load_image_from_path(p)
            frame = frame_option.unwrap()
            if frame is None:
                actual.write(f"An error has occured, please try again with another image. ERROR: {frame_option.logs}")
                return    


            t1 = time()
            data_mp = get_mediapipe_prediction(frame)

            t2 = time()
            data_mn = get_movenet_prediction(frame)
            t3 = time()

            draw_rect(frame, 
                    data_mp, 
                    t2-t1,
                    prediction_mp, 
                    imageLocation_mp)  

            draw_rect(frame, 
                    data_mn, 
                    t3-t2, 
                    prediction_mn,
                    imageLocation_mn
                    )  

            
        except Exception as e: 
            st.write(e)
            
    return predict


def evaluate_iou(pred, title="title", start=4, end=8):
    res = []
    for i in range(start, end):
        iou = i/10
        res.append([iou, np.mean(pred>=iou)])
    res = np.array(res)
    fig, ax = plt.subplots()
    ax.plot(res[:, 0], res[:, 1])
    ax.set_title(title)
    ax.set_xlabel("iou")
    ax.set_ylabel("score")
    ax.set_ylim([0, 1])
    ax.grid(1)
    return fig    

# %%

st.title("""
Introduction
""")

st.markdown(
"""
<ul>
<li><b>Face Detection</b> using <a href="https://www.tensorflow.org/hub/tutorials/movenet#:~:text=MoveNet%20is%20an%20ultra%20fast,applications%20that%20require%20high%20accuracy.">movenet multipose</a> and <a href="https://google.github.io/mediapipe/solutions/face_detection.html">mediapipe face detection</a></li>
<div stle="display:flex;">
<img src="https://google.github.io/mediapipe/images/mobile/face_detection_android_gpu.gif" height=200/>
<img src="https://1.bp.blogspot.com/-z7eLvmyTc6Y/YJ6y4qWlW0I/AAAAAAAAEM0/GhsdUgw8dQk8zF1G4rXukd2PlCtGJ5PHACLcBGAsYHQ/s0/anastasia_labeled.jpeg" height=200/>
</div>
<li><b>Gender detection</b> and <b>Age estimation</b> using <a href="https://becominghuman.ai/detecting-age-and-gender-with-tf-lite-on-android-33997eed6c25">this model</a> </li>
<img src="https://raw.githubusercontent.com/shubham0204/Age-Gender_Estimation_TF-Android/master/images/results.png" width=800/>

</ul>    
""" , unsafe_allow_html=True)

st.title("""
Demo
""")
path = st.text_input('Enter The path of a local image or a url', "")
predict_1 = setup_prediction()

if path!="":
   predict_1(path)


st.title("""
Evaluation
""")

st.markdown("""
<li>
<strong>Face detection IOU evaluation using <a href="http://shuoyang1213.me/WIDERFACE/">Wider Face Dataset</a> (Images containing 1  person)</strong>
</li>
<br/>
""", unsafe_allow_html=True)

st.markdown(f"""
    <img src="https://www.researchgate.net/publication/335876570/figure/fig2/AS:804291526795265@1568769451765/Intersection-over-Union-IOU-calculation-diagram.png" width=500 />
    <br/>
    <br/>

    """, unsafe_allow_html=True)

wider_val_mp, wider_val_mn = np.load("results/wider_face_val_bbx_gt.npy").transpose()
col1, col2 = st.columns(2)
col1.pyplot(evaluate_iou(wider_val_mn, title="movenet evaluation on wider_face dataset"))
col2.pyplot(evaluate_iou(wider_val_mp, title="mediapipe evaluation on wider_face dataset"))

st.markdown("""
<li>
Age and Gender detection, accuracy and mae evaluation using <a href="https://susanqq.github.io/UTKFace/">UTKFace Dataset</a>
</li>
<br/>

<img src="https://i.imgur.com/19LNbyQ.jpg" width=500/>
<img src="https://miro.medium.com/max/1400/1*udGMH6OQF4CMcv42mjW_qg.png" width=500/>
<br/>

<ol> <h2> Method </h2>
<li>Use a Face detection model to detect face bounding box from image</li>
<li>Use predicted bounding box to crop the image resulting in a copped image containing only the face of the target</li>
<li>Use a age estimation and gender detection model to predict the age and gender of the face in the cropped image</li>
</ol>
<br/>

""", unsafe_allow_html=True)
col1, col2 = st.columns(2)

results = np.load("results/UTK_part1.npy")

mn = np.mean(results[:, 1, 0]), np.mean(np.abs(results[:, 1, 1]))
mp = np.mean(results[:, 0, 0]), np.mean(np.abs(results[:, 0, 1]))

df = pd.DataFrame(
    np.array([["mediapipe",np.round(mp[0], 2), np.round(mp[1], 2)], ["movenet", np.round(mn[0], 2), np.round(mn[1], 2)]]),
    columns=("Model", "gender accuracy", "age mae"))
st.markdown("""
<p>Movenet vs Mediapipe Age and gender detection Results</p>
""", unsafe_allow_html=True)
st.table(df)