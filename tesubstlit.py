from turtle import width
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import streamlit as st
from PIL import Image
import base64


model = load_model('model_1.h5')
Classes = ['Bacterial Blight','Blast','Brown spot','Healthy Crop','Hispa','Leaf Blast','Tungro']

header = st.container()
section = st.container()
detection = st.container()

@st.cache
def load_image(file):
    img = Image.open(file)
    return img



with header:
    st.title("RICE CROP🌱 DISEASE DETECTION")
    file = st.file_uploader("", type=['jpg','png', 'jpeg'])

with section:
    if file is None:
        st.subheader('Upload image👆')
        
    if file is not None:
        
        with open(os.path.join('C:/Users/Sahil Nagpure/Projects/Deep Learning/',file.name),'wb') as f:
            f.write(file.getbuffer())
            st.success("image uploaded successfully")
        img = load_image(file)
        st.image(img, width=200)
        

        img=image.load_img(os.path.join('C:/Users/Sahil Nagpure/Projects/Deep Learning/',file.name),target_size=(96,96,3))
        img=image.img_to_array(img)
        img=np.expand_dims(img,axis=0)


        predimg = model.predict(img)
        predindex = np.argmax(predimg)
        pred = Classes[predindex]
        if pred =='Healthy Crop':
            st.markdown(f"<h4 style='text-align:left'>{pred}</h4>", unsafe_allow_html=True)


with detection:
    if file is not None:
        if pred!='Healthy Crop':
            st.subheader("**Type of Disease:**")
            st.markdown(f"<h4 style='text-align:left'>{pred}</h4>", unsafe_allow_html=True)

