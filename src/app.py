import os

import numpy as np
import json
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from flask import Flask, url_for, request, render_template, redirect
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename


app = Flask(__name__)


root_dir = os.getcwd()
model_dir_path = os.path.join(root_dir, 'models')
config_file_name = 'pneumonia_detection_inception_model.json'
weights_file_name = 'pneumonia_detection_inception_model_03-0.845000.h5'

xray_result = ['NORMAL', 'PNEUMONIA']

json_file = open(os.path.join(model_dir_path, config_file_name),'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights(os.path.join(model_dir_path, weights_file_name))


def model_predict(model, image_path):

    img = image.load_img(image_path,target_size=(400,400))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = x/255.0
    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def index():

    return render_template('./index.html')

@app.route('/',methods=['GET','POST'])
def upload():

    if request.method=='POST':
        f = request.files['file']
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path,'uploads',secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(model, file_path)
        if preds > 0.5:
            pred_class = 1
        else:
            pred_class = 0
        result = xray_result[pred_class]
        return render_template('./predict.html',result=result)
    else:
        return render_template('./index.html')

if __name__ == '__main__':
    app.run(debug=True)
