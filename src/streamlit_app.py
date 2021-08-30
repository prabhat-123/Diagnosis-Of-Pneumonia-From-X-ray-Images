import os

import json
import numpy as np
from PIL import Image
import streamlit as st
from predictor import Predictor
from tensorflow.keras.models import model_from_json



xray_types = ['NORMAL', 'PNEUMONIA']
predictor = Predictor(model_dir_name = 'models')
loaded_model_json = predictor.load_config(config_file_name = 'pneumonia_detection_xception_model.json')
model = predictor.load_weights(weights_file_name = 'pneumonia_detection_model_01-0.835000.h5', loaded_model_json = loaded_model_json)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Chest Xray Image Classification Example")
st.header("Identifying Pneumonia Cases From Chest Xrays")
st.text("Upload a chest X-ray Image for image classification as pneumonia or normal")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'uploads', uploaded_file.name)
    img.save(file_path)
    st.image(img, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second...")
    label = predictor.predict(model, file_path)
    if label == 0:
        st.write("The chest xray of a patient has a Pneumonia.")
    else:
        st.write("The chest xray of a patient is Normal")





