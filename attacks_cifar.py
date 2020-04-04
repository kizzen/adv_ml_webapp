from art.attacks import DeepFool, FastGradientMethod, SaliencyMapMethod
from art.classifiers import KerasClassifier
import foolbox
from foolbox.attacks import FGSM
from foolbox.criteria import Misclassification
from diffimg import diff
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import model_from_json
import os

# hyperparam tuning
batch_size = 32
num_classes = 10
epochs = 100
num_predictions = 20

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# calculate accuracy of the models
# model loading/reconstruction from JSON file  
with open('flask/static/CIFAR/models/CNN_architecture.json', 'r') as f:
    model = model_from_json(f.read())  
# load weights into the new model
model.load_weights('flask/static/CIFAR/models/CNN_weights.h5')

classes_mapping = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
#functions to generate attacks and store adverarial images
# FGSM
def store_images_fgsm(num):
	# take an image
	img_num = num
	img = x_test[img_num] 
	# take its label
	true_label = classes_mapping[y_test[img_num].argmax()]
	# predict it using CNN
	predict_original = classes_mapping[model.predict_classes(img.reshape(1,32,32,3))[0]]
	# save image to be displayed on the screen
	plt.title('Original Image', fontsize = 20)
	plt.xlabel('True Class: {} \n {}: {}'.format(true_label,'CNN Prediction',predict_original),fontsize=15) 
	# plot it
	fig = plt.gcf()
	plt.imshow(img.reshape((32,32,3)), cmap='Greys') #plot it
	plt.tight_layout()
	# save original image
	path = 'images/CIFAR/FGSM/'
	filename_original = 'imgnum' + str(img_num) + '_label' + str(true_label) + '_pred' + str(predict_original) + '.png'
	fig.savefig(path + filename_original,bbox_inches='tight')
	# get it's adversarial example
	classifier = KerasClassifier(clip_values=(0, 255), model=model)
	epsilon = 0.2
	adv_crafter = FastGradientMethod(classifier)
	x_art = np.reshape(img,[1,32,32,3])
	img_adv = adv_crafter.generate(x=x_art, eps=epsilon)
	# get CNN prediction on adversarial example
	predict_adv = classes_mapping[model.predict_classes(img_adv.reshape(1,32,32,3))[0]]
	filename_adversarial_diff = 'advimgnum' + str(img_num) + '_noiseNA' + '_pred' + predict_adv + '.png'
	# get noise different between original image and adversarial example
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((32,32,3)), cmap='Greys') #plot it
	plt.tight_layout()
	fig.savefig(path + filename_adversarial_diff,bbox_inches='tight')
	# save image to be displayed on the screen
	im_diff = round(diff(path + filename_original, path +  filename_adversarial_diff) * 100,2)
	plt.title('{} Attack'.format('FGSM'), fontsize = 20)
	plt.xlabel('Noise level: {}% \n {}: {}'.format(im_diff,'CNN Prediction',predict_adv),fontsize=15)
	# plot it
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((32,32,3)), cmap='Greys') #plot it
	plt.tight_layout()
	# save adversarial image
	filename_adversarial = 'advimgnum' + str(img_num) + '_noise' + str(im_diff) + '_pred' + predict_adv + '.png'
	fig.savefig(path + filename_adversarial,bbox_inches='tight')
	os.remove(path + filename_adversarial_diff)
	if im_diff == 0.0:
		os.remove(path + filename_adversarial)
		os.remove(path + filename_original)
	else:
		pass

# JSMA
def store_images_jsma(num):
	# take an image
	img_num = num
	img = x_test[img_num] 
	# take its label
	true_label = classes_mapping[y_test[img_num].argmax()]
	# predict it using CNN
	predict_original = classes_mapping[model.predict_classes(img.reshape(1,32,32,3))[0]]
	# save image to be displayed on the screen
	plt.title('Original Image', fontsize = 20)
	plt.xlabel('True Class: {} \n {}: {}'.format(true_label,'CNN Prediction',predict_original),fontsize=15) 
	# plot it
	fig = plt.gcf()
	plt.imshow(img.reshape((32,32,3)), cmap='Greys') #plot it
	plt.tight_layout()
	# save original image
	path = 'images/CIFAR/JSMA/'
	filename_original = 'imgnum' + str(img_num) + '_label' + str(true_label) + '_pred' + predict_original + '.png'
	fig.savefig(path + filename_original,bbox_inches='tight')
	# get it's adversarial example
	classifier = KerasClassifier(clip_values=(0, 255), model=model)
	epsilon = 0.2
	adv_crafter = SaliencyMapMethod(classifier)
	x_art = np.reshape(img,[1,32,32,3])
	img_adv = adv_crafter.generate(x=x_art, eps=epsilon)
	# get CNN prediction on adversarial example
	predict_adv = classes_mapping[model.predict_classes(img_adv.reshape(1,32,32,3))[0]]
	filename_adversarial_diff = 'advimgnum' + str(img_num) + '_noiseNA' + '_pred' + predict_adv + '.png'
	# get noise different between original image and adversarial example
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((32,32,3)), cmap='Greys') #plot it
	plt.tight_layout()
	fig.savefig(path + filename_adversarial_diff,bbox_inches='tight')
	# save image to be displayed on the screen
	im_diff = round(diff(path + filename_original, path +  filename_adversarial_diff) * 100,2)
	plt.title('{} Attack'.format('JSMA'), fontsize = 20)
	plt.xlabel('Noise level: {}% \n {}: {}'.format(im_diff,'CNN Prediction',predict_adv),fontsize=15)
	# plot it
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((32,32,3)), cmap='Greys') #plot it
	plt.tight_layout()
	# save adversarial image
	filename_adversarial = 'advimgnum' + str(img_num) + '_noise' + str(im_diff) + '_pred' + predict_adv + '.png'
	fig.savefig(path + filename_adversarial,bbox_inches='tight')
	os.remove(path + filename_adversarial_diff)
	if im_diff == 0.0:
		os.remove(path + filename_adversarial)
		os.remove(path + filename_original)
	else:
		pass

# DeepFool
def store_images_deepfool(num):
	# take an image
	img_num = num
	img = x_test[img_num] 
	# take its label
	true_label = classes_mapping[y_test[img_num].argmax()]
	# predict it using CNN
	predict_original = classes_mapping[model.predict_classes(img.reshape(1,32,32,3))[0]]
	# save image to be displayed on the screen
	plt.title('Original Image', fontsize = 20)
	plt.xlabel('True Class: {} \n {}: {}'.format(true_label,'CNN Prediction',predict_original),fontsize=15) 
	# plot it
	fig = plt.gcf()
	plt.imshow(img.reshape((32,32,3)), cmap='Greys') #plot it
	plt.tight_layout()
	# save original image
	path = 'images/CIFAR/DeepFool/'
	filename_original = 'imgnum' + str(img_num) + '_label' + str(true_label) + '_pred' + predict_original + '.png'
	fig.savefig(path + filename_original,bbox_inches='tight')
	# get it's adversarial example
	classifier = KerasClassifier(clip_values=(0, 255), model=model)
	epsilon = 0.2
	adv_crafter = DeepFool(classifier)
	x_art = np.reshape(img,[1,32,32,3])
	img_adv = adv_crafter.generate(x=x_art, eps=epsilon)
	# get CNN prediction on adversarial example
	predict_adv = classes_mapping[model.predict_classes(img_adv.reshape(1,32,32,3))[0]]
	filename_adversarial_diff = 'advimgnum' + str(img_num) + '_noiseNA' + '_pred' + predict_adv + '.png'
	# get noise different between original image and adversarial example
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((32,32,3)), cmap='Greys') #plot it
	plt.tight_layout()
	fig.savefig(path + filename_adversarial_diff,bbox_inches='tight')
	# save image to be displayed on the screen
	im_diff = round(diff(path + filename_original, path +  filename_adversarial_diff) * 100,2)
	plt.title('{} Attack'.format('DeepFool'), fontsize = 20)
	plt.xlabel('Noise level: {}% \n {}: {}'.format(im_diff,'CNN Prediction',predict_adv),fontsize=15)
	# plot it
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((32,32,3)), cmap='Greys') #plot it
	plt.tight_layout()
	# save adversarial image
	filename_adversarial = 'advimgnum' + str(img_num) + '_noise' + str(im_diff) + '_pred' + str(predict_adv) + '.png'
	fig.savefig(path + filename_adversarial,bbox_inches='tight')
	os.remove(path + filename_adversarial_diff)
	if im_diff == 0.0:
		os.remove(path + filename_adversarial)
		os.remove(path + filename_original)
	else:
		pass
num_store_image = 200
for i in range(150,num_store_image):
	store_images_fgsm(i)
	print('FGSM iteration:', i)
for i in range(150,num_store_image):
	store_images_jsma(i)
	print('JSMA iteration:', i)
for i in range(150,num_store_image):
	store_images_deepfool(i)
	print('DeepFool iteration:', i)
	
