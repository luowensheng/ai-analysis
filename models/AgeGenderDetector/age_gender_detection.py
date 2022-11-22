#%%
from keras.models import load_model
import tensorflow as tf
import cv2

model_path = "./weights/age_gender_model2.h5"
model = load_model(model_path)
model.compile()
MAX_AGE = 116



def preprocess(image, shape=(200, 200)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
    image = tf.convert_to_tensor(image/255, dtype=tf.float32)
    return tf.expand_dims(image, 0)


def predict(x):
    pred = model.predict(preprocess(x))[0]
    age_pred = int(pred[0]*MAX_AGE)
    # gender_pred = "male" if int(pred[1].argmax(1))==0 else "female"
    gender_pred = int(pred[1:].argmax())
    return {"age": age_pred, "gender":gender_pred}
    


# %%
