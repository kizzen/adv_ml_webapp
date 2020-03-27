from flask import Flask, render_template, request, url_for 
import os
from os import listdir
import pandas as pd
import numpy as np
from PIL import Image
import random
import time
import keras
from keras.models import load_model
from keras import backend as K

app = Flask(__name__)

@app.route('/',methods=['POST', 'GET'])
@app.route('/advml_home',methods=['POST', 'GET'])
def advml_home():
	return render_template('advml_home.html')

@app.route('/advml_pics',methods=['POST', 'GET'])
def advml_pics():
	dataset_select = 'MNIST'
	# dataset selection from UI
	if dataset_select == 'MNIST':
		path_models_MNIST = '/Users/khalilezzine/Desktop/DS/portfolio/adversarial_ml/models/MNIST/'
		# load models
		# model reconstruction from JSON file
		with open(path + 'CNN_architecture.json', 'r') as f:
		    model = model_from_json(f.read())
		# load weights into the new model
		model.load_weights(path + 'CNN_weights.h5')
		keras.backend.clear_session() 

		dataset_select = request.form.get("dataset")
		model_select = request.form.get("model")
		if dataset_select == 'MNIST':
			dataset = 'MNIST'
			dataset_type = 'MNIST CNN Prediction'
		elif dataset_select == 'CIFAR':
			dataset = 'CIFAR'
			dataset_type = 'CIFAR CNN Prediction'
		# else statement for when app first open and no selection made (random selection)
		else:
			dataset_select = random.choice(['MNIST','CIFAR'])
			if dataset_select == 'MNSIT':
				dataset = 'MNIST'
				dataset_type = 'MNIST CNN Prediction'
			elif dataset_select == 'CIFAR':
				dataset = 'CIFAR'
				cnn_type = 'CIFAR CNN Prediction'
		# attack selection
		attack_select = request.form.get("attack")
		if attack_select == 'fgsm':
			attack_type = 'FGSM'
		elif attack_select == 'jsma':
			attack_type = 'JSMA'
		elif attack_select == 'deepfool':
			attack_type = 'DeepFool'
		else:
			attack_select = random.choice(['fgsm','jsma','deepfool'])
			if attack_select == 'fgsm':
				attack_type_leg = 'FGSM'
			elif attack_select == 'jsma':
				attack_type_leg = 'JSMA'
			elif attack_select == 'deepfool':
				attack_type_leg = 'DeepFool'

#Caching
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == '__main__':
   	app.run(host='0.0.0.0', port=80)