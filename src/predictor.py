import os

import json
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

class Predictor():

    def __init__(self, model_dir_name):

        self.model_dir_name = model_dir_name
        root_dir = os.getcwd()
        model_dir_path = os.path.join(root_dir , model_dir_name)
        

    
    def load_config(self, config_file_name):
        
        self.config_file_name = config_file_name
        json_file = open(os.path.join(self.model_dir_name, self.config_file_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        return loaded_model_json


    def load_weights(self, weights_file_name, loaded_model_json):

        self.weights_file_name = weights_file_name
        self.loaded_model_json = loaded_model_json
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(self.model_dir_name, weights_file_name))
        return model

    


    def predict(self, model, img_path):

        self.model = model
        self.img_path = img_path
        img = image.load_img(img_path, target_size=(400,400))
        img_arr = image.img_to_array(img)
        normalized_img_arr = img_arr/255.
        test_img = np.expand_dims(normalized_img_arr, axis = 0)
        preds = model.predict(test_img)
        return preds
