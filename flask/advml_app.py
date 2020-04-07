from flask import Flask, render_template, request, url_for 
import os
from os import listdir
import random
import time
import keras
from keras.models import load_model
from keras.models import model_from_json
from keras import backend as K

app = Flask(__name__)

@app.route('/',methods=['POST', 'GET'])
def advml_home():
	return render_template('advml_home.html')

@app.route('/advml_pics',methods=['POST', 'GET'])
def advml_pics():
	dataset_select = request.form.get("dataset")
	if dataset_select == 'MNIST':
		dataset = 'MNIST'
	elif dataset_select == 'CIFAR':
		dataset = 'CIFAR'
	else:
		dataset = random.choice(['MNIST','CIFAR'])
	# load model
	path_model = os.getcwd() + '/static/' + dataset + '/models/' 
	with open(path_model + 'CNN_architecture.json', 'r') as f:
		model = model_from_json(f.read())
	model.load_weights(path_model + 'CNN_weights.h5')

	# attack selection
	attack_select = request.form.get("attack")
	if attack_select == 'FGSM':
		attack = 'FGSM'
	elif attack_select == 'JSMA':
		attack = 'JSMA'
	elif attack_select == 'DeepFool':
		attack = 'DeepFool'
	else:
		attack = random.choice(['FGSM','JSMA','DeepFool'])
	
	# randomly choose an image	
	img_path = 'static/' + dataset + '/' + attack + '/' 
	img_fname = random.choice([f for f in os.listdir(img_path) if f.startswith('imgnum')])
	# find the adv example
	adv_img_num = img_fname.split('_')[0].split('num')[1]
	for f in os.listdir(img_path):
		if f.startswith('advimgnum' + adv_img_num + '_'):
			advimg_fname = f

	print('img: ',img_fname)	
	print('advimg: ',advimg_fname)	

	keras.backend.clear_session() 
	return render_template('advml_pics.html', img_clean = img_path + img_fname, img_adv=img_path + advimg_fname)

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
	app.run(host='0.0.0.0',port=80,debug=True)
