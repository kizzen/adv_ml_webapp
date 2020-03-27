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
from keras.datasets import mnist
from keras.models import model_from_json

# hyperparam tuning
img_rows, img_cols = 28, 28 # image dimensions
channels=1 # channel for black and white
num_classes = 10 # 0 through 9 digits as class
params = [32, 32, 64, 64, 200, 200] # parameter for the CNN
batch_size = 128 # batch size

# loading mnist data from keras
# load and split data between test and train set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data transformation for model
from keras import backend as K
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model loading/reconstruction from JSON file  
with open('models/MNIST/CNN_architecture.json', 'r') as f:
    model = model_from_json(f.read())  
# load weights into the new model
model.load_weights('models/MNIST/CNN_weights.h5')

#functions to generate attacks and store adverarial images
# FGSM
def store_images_fgsm(num):
	# take an image
	img_num = num
	img = x_test[img_num] 
	# take its label
	true_label = y_test[img_num].argmax()
	# predict it using CNN
	predict_original = model.predict_classes(img.reshape(1,28,28,1))[0]
	# save image to be displayed on the screen
	plt.title('Original Image', fontsize = 20)
	plt.xlabel('True Class: {} \n {}: {}'.format(true_label,'CNN Prediction',predict_original),fontsize=15) 
	# plot it
	fig = plt.gcf()
	plt.imshow(img.reshape((28,28)), cmap='Greys') #plot it
	plt.tight_layout()
	# save original image
	path = 'images/MNIST/FGSM/'
	filename_original = 'imgnum' + str(img_num) + '_label' + str(true_label) + '_pred' + str(predict_original) + '.png'
	fig.savefig(path + filename_original,bbox_inches='tight')
	# get it's adversarial example
	classifier = KerasClassifier(clip_values=(0, 255), model=model)
	epsilon = 0.2
	adv_crafter = FastGradientMethod(classifier)
	x_art = np.reshape(img,[1,28,28,1])
	img_adv = adv_crafter.generate(x=x_art, eps=epsilon)
	# get CNN prediction on adversarial example
	predict_adv = model.predict_classes(img_adv.reshape(1,28,28,1))[0]
	filename_adversarial_diff = 'advimgnum' + str(img_num) + '_noiseNA' + '_pred' + str(predict_adv) + '.png'
	# get noise different between original image and adversarial example
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((28,28)), cmap='Greys') #plot it
	plt.tight_layout()
	fig.savefig(path + filename_adversarial_diff,bbox_inches='tight')
	# save image to be displayed on the screen
	im_diff = round(diff(path + filename_original, path +  filename_adversarial_diff) * 100,2)
	plt.title('{} Adversarial Image'.format('FGSM'), fontsize = 20)
	plt.xlabel('Noise level: {}% \n {}: {}'.format(im_diff,'CNN Prediction',predict_adv),fontsize=15)
	# plot it
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((28,28)), cmap='Greys') #plot it
	plt.tight_layout()
	# save adversarial image
	filename_adversarial = 'advimgnum' + str(img_num) + '_noise' + str(im_diff) + '_pred' + str(predict_adv) + '.png'
	fig.savefig(path + filename_adversarial,bbox_inches='tight')
	os.remove(path + filename_adversarial_diff)

# JSMA
def store_images_jsma(num):
	# take an image
	img_num = num
	img = x_test[img_num] 
	# take its label
	true_label = y_test[img_num].argmax()
	# predict it using CNN
	predict_original = model.predict_classes(img.reshape(1,28,28,1))[0]
	# save image to be displayed on the screen
	plt.title('Original Image', fontsize = 20)
	plt.xlabel('True Class: {} \n {}: {}'.format(true_label,'CNN Prediction',predict_original),fontsize=15) 
	# plot it
	fig = plt.gcf()
	plt.imshow(img.reshape((28,28)), cmap='Greys') #plot it
	plt.tight_layout()
	# save original image
	path = 'images/MNIST/JSMA/'
	filename_original = 'imgnum' + str(img_num) + '_label' + str(true_label) + '_pred' + str(predict_original) + '.png'
	fig.savefig(path + filename_original,bbox_inches='tight')
	# get it's adversarial example
	classifier = KerasClassifier(clip_values=(0, 255), model=model)
	epsilon = 0.2
	adv_crafter = SaliencyMapMethod(classifier)
	x_art = np.reshape(img,[1,28,28,1])
	img_adv = adv_crafter.generate(x=x_art, eps=epsilon)
	# get CNN prediction on adversarial example
	predict_adv = model.predict_classes(img_adv.reshape(1,28,28,1))[0]
	filename_adversarial_diff = 'advimgnum' + str(img_num) + '_noiseNA' + '_pred' + str(predict_adv) + '.png'
	# get noise different between original image and adversarial example
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((28,28)), cmap='Greys') #plot it
	plt.tight_layout()
	fig.savefig(path + filename_adversarial_diff,bbox_inches='tight')
	# save image to be displayed on the screen
	im_diff = round(diff(path + filename_original, path +  filename_adversarial_diff) * 100,2)
	plt.title('{} Adversarial Image'.format('JSMA'), fontsize = 20)
	plt.xlabel('Noise level: {}% \n {}: {}'.format(im_diff,'CNN Prediction',predict_adv),fontsize=15)
	# plot it
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((28,28)), cmap='Greys') #plot it
	plt.tight_layout()
	# save adversarial image
	filename_adversarial = 'advimgnum' + str(img_num) + '_noise' + str(im_diff) + '_pred' + str(predict_adv) + '.png'
	fig.savefig(path + filename_adversarial,bbox_inches='tight')
	os.remove(path + filename_adversarial_diff)

# DeepFool
def store_images_deepfool(num):
	# take an image
	img_num = num
	img = x_test[img_num] 
	# take its label
	true_label = y_test[img_num].argmax()
	# predict it using CNN
	predict_original = model.predict_classes(img.reshape(1,28,28,1))[0]
	# save image to be displayed on the screen
	plt.title('Original Image', fontsize = 20)
	plt.xlabel('True Class: {} \n {}: {}'.format(true_label,'CNN Prediction',predict_original),fontsize=15) 
	# plot it
	fig = plt.gcf()
	plt.imshow(img.reshape((28,28)), cmap='Greys') #plot it
	plt.tight_layout()
	# save original image
	path = 'images/MNIST/DeepFool/'
	filename_original = 'imgnum' + str(img_num) + '_label' + str(true_label) + '_pred' + str(predict_original) + '.png'
	fig.savefig(path + filename_original,bbox_inches='tight')
	# get it's adversarial example
	classifier = KerasClassifier(clip_values=(0, 255), model=model)
	epsilon = 0.2
	adv_crafter = DeepFool(classifier)
	x_art = np.reshape(img,[1,28,28,1])
	img_adv = adv_crafter.generate(x=x_art, eps=epsilon)
	# get CNN prediction on adversarial example
	predict_adv = model.predict_classes(img_adv.reshape(1,28,28,1))[0]
	filename_adversarial_diff = 'advimgnum' + str(img_num) + '_noiseNA' + '_pred' + str(predict_adv) + '.png'
	# get noise different between original image and adversarial example
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((28,28)), cmap='Greys') #plot it
	plt.tight_layout()
	fig.savefig(path + filename_adversarial_diff,bbox_inches='tight')
	# save image to be displayed on the screen
	im_diff = round(diff(path + filename_original, path +  filename_adversarial_diff) * 100,2)
	plt.title('{} Adversarial Image'.format('DeepFool'), fontsize = 20)
	plt.xlabel('Noise level: {}% \n {}: {}'.format(im_diff,'CNN Prediction',predict_adv),fontsize=15)
	# plot it
	fig = plt.gcf()
	plt.imshow(img_adv.reshape((28,28)), cmap='Greys') #plot it
	plt.tight_layout()
	# save adversarial image
	filename_adversarial = 'advimgnum' + str(img_num) + '_noise' + str(im_diff) + '_pred' + str(predict_adv) + '.png'
	fig.savefig(path + filename_adversarial,bbox_inches='tight')
	os.remove(path + filename_adversarial_diff)

count = 0
for i in range(0,y_test.shape[0]-1):
	try:
		store_images_fgsm(i)
		store_images_jsma(i)
		store_images_deepfool(i)
	except:
		pass
	count +=1
	print('iteration:', count)
