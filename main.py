from ast import While
from time import sleep, time

import streamlit as st
from prediction import setup_prediction, setup_video
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataset import dataset
from collect import collect_img_from_google_search
import streamlit.components.v1 as components

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
def stringify_attributes(attributes={}):
    return ' '.join([f'{attr}="{attributes[attr]}"' for attr in attributes])

def add_content(items, attributes={}, extra=''):
    return f"{extra} <ul>"+' '.join([f"<li {stringify_attributes(attributes)}>{item}</li>" for item in items]) + "</ul>" 

add_listitem = lambda items, extra='': add_content(items, {'style':'padding-right:10px; padding-bottom:10px; font-family:"Poppins";'}, extra=extra)

wrap_title = lambda content:f"* <h3><b>{content}</b><h3>"

def add_resource(url, title, content, img_src, img_width=100, img_height=200):
    return f"""
                    <div style="display:grid; grid-template-columns: repeat(9, 1fr); margin-bottom:20px" >        
                    <div style=" grid-column: 1/ 4; padding: 5px;">
                        <a  href="{url}"> {title}</a> 
                        <p style="overflow-wrap: break-word; word-wrap: break-word; hyphens: auto;">
                            {content}
                        </p>                        
                    </div>

                    <img src="{img_src}" 
                        width={img_width} height={img_height} style="grid-column: 4/ -1;"/>
                </div>
    """

def wrap_with_html_tag(tag, content, attributes={}):
    return f"""<{tag} {stringify_attributes(attributes)}>
             {content}
            </{tag}>
           """


st.title("""
Introduction
Goal: Build a fast, lightweight accurate model capable of determing the age and gender of users using face detection and age and gender detection models.  
""")


components.html(
f"""
    <link href="https://font.googleapis.com/css?family=Poppins" rel="stylesheet">
    
    <ol>
        <li> {wrap_with_html_tag('h3', 
               wrap_with_html_tag('b', "Face Detection Models"))}
        
        {add_resource(
            url="https://google.github.io/mediapipe/solutions/face_detection.html", 
            title="Mediapipe face detection", 
            content=add_listitem([
                "Fast", 
                "lightweight", 
                "Optimized for cpu and mobile devices",
                wrap_with_html_tag("a", 
                                   "Paper-- BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs", 
                                   {'href':"https://arxiv.org/pdf/1907.05047.pdf"})
            ]),
            img_src="https://google.github.io/mediapipe/images/mobile/face_detection_android_gpu.gif",
            img_width=150, 
            img_height=300
        )}

        {add_resource(
            url="https://www.tensorflow.org/hub/tutorials/movenet#:~:text=MoveNet%20is%20an%20ultra%20fast,applications%20that%20require%20high%20accuracy", 
            title="Movenet Multipose", 
            content=add_listitem([
                "Fast and lightweight", 
                "Optimized for cpu and mobile devices",
                "Used for both single and multiple person pose estimation"
            ]),
            img_src="https://1.bp.blogspot.com/-z7eLvmyTc6Y/YJ6y4qWlW0I/AAAAAAAAEM0/GhsdUgw8dQk8zF1G4rXukd2PlCtGJ5PHACLcBGAsYHQ/s0/anastasia_labeled.jpeg",
            img_width=300, 
            img_height=300
        )}

        {add_resource(
            url="https://github.com/ipazc/mtcnn", 
            title="MTCNN face detection", 
            content=add_listitem([
                "Fast and lightweight", 
                wrap_with_html_tag('a', 
                                   "Paper-- Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Network", 
                                   {'href':"https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf"})
            ]),
            img_src="https://camo.githubusercontent.com/8278dc58627991ed14aa3a0a9c1c127635b6c2eac2264e61fec4789d41ad6e85/68747470733a2f2f6b707a68616e6739332e6769746875622e696f2f4d54434e4e5f666163655f646574656374696f6e5f616c69676e6d656e742f70617065722f6578616d706c65732e706e67",
            img_width=550, 
            img_height=250
        )}                
       
        </li>

        <li> {wrap_with_html_tag('h3', 
               wrap_with_html_tag('b', "Age Gender Estimation Models"))}

           {add_resource(
            url="https://becominghuman.ai/detecting-age-and-gender-with-tf-lite-on-android-33997eed6c25", 
            title="Ligthweight and fast age gender Estimation model", 
            content=add_listitem([
                "Fast and lightweight",
                f"Has two options: <br>{add_listitem([add_listitem(['slower', 'Better performance'], extra='Vanilla: '), add_listitem(['Faster', 'Better optimized for mobile devices'], extra='Lite:')])}",  
                wrap_with_html_tag('a', 
                                   "Code", 
                                   {'href':"https://github.com/shubham0204/Age-Gender_Estimation_TF-Android"})
            ]),
            img_src="https://raw.githubusercontent.com/shubham0204/Age-Gender_Estimation_TF-Android/master/images/results.png",
            img_width=550, 
            img_height=500
        )}  
        
        </li>
    </ol> 


""", width=900, height=1600, scrolling=True )

st.markdown("""---""") 
st.markdown("""---""") 

st.title("""
Demo
""")



st.markdown(f"""
{wrap_title("Demo 1/4")}
Using google image search to get images 
""", unsafe_allow_html=True)



search = st.text_input('Search a person', "tsai ying wen")
clicked = st.button("Predict Google search")
# next_image = st.empty()
predict_on_search = setup_prediction()

def predict_from_url(search):
    urls = collect_img_from_google_search(search)[:3]
    for i, url in enumerate(urls):
          setup_prediction()(url)


if clicked:
    if search!="":
        predict_from_url(search)

else:
    if search=="" or search=="tsai ying wen":
       predict_from_url("tsai ying wen")

sleep(1)

st.markdown("""---""")
st.markdown(f"""
{wrap_title("Demo 2/4")}
Using online and local or images 
""", unsafe_allow_html=True)
path = st.text_input('Enter the full path of a local image or a url', "https://tnimage.s3.hicloud.net.tw/photos/2019/07/22/1563786906-5d357e9a8b721.jpg")
clicked = st.button("Predict")

predict_1 = setup_prediction()


if clicked:
    if path!="":
        predict_1(path)

else:
   predict_1(path)

st.markdown("""---""")
st.markdown(f"""
{wrap_title("Demo 3/4")}
Using images from UTK dataset 
click the "generate" button bellow to view predictions
""", unsafe_allow_html=True)

option_age = st.selectbox('select an age', tuple(["None"]+list(range(116))))

option_gender = st.selectbox('Select a gender', ("None", 'male', 'female'))


clicked = st.button("Generate")

predict = setup_prediction()



if clicked:
    p = dataset.by(age=option_age, gender=option_gender)   
    predict(p)

else:
    p = dataset.by()
    predict(p)


st.markdown("""---""")
st.markdown(f"""
{wrap_title("Demo 4/4")}
Using videos from a local folder
click the "generate" button bellow to view predictions
""", unsafe_allow_html=True)

video = "C:/....*/mp4"

video_path = st.text_input('Enter The path of a local video or a url', video)

clicked = st.button("Video Demo")


if clicked:
    if video_path!='' and video_path!=video:
        play_video = setup_video()
        play_video(video_path)

st.markdown("""---""") 
st.markdown("""---""")
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
""", unsafe_allow_html=True)
st.markdown("""---""") 
st.markdown("""---""")

st.title("Method")
st.markdown("""
<ol>
<li>Use a Face detection model to detect face bounding box from image</li>
<li>Use predicted bounding box to crop the image resulting in a copped image containing only the face of the target</li>
<li>Use an age estimation and gender detection model to predict the age and gender of the face in the cropped image</li>
</ol>
<br/>

""", unsafe_allow_html=True)
st.markdown("""---""") 
st.markdown("""---""") 

st.title("Results")
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

st.markdown("""---""") 
st.markdown("""---""") 
# %%
