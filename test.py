from art.attacks import DeepFool, FastGradientMethod, CarliniL2Method, SaliencyMapMethod, BasicIterativeMethod
from art.classifiers import KerasClassifier
import foolbox
from foolbox.attacks import FGSM
from foolbox.criteria import Misclassification

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from diffimg import diff
import os

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

# function to train CNN model
def train(filename,filename_weights,filename_archi,params,epochs=50,
          batch_size=128,train_temp=1,init=None):

    # build CNN model
    model = Sequential()
    model.add(Conv2D(params[0], (3, 3),input_shape=(img_rows, img_cols,channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))

    if init != None:
        model.load_weights(init)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=epochs,
              shuffle=True)
    
    model.save(filename) # save entire model
    model.save_weights(filename_weights) # save model weight to the filename param in the function  
    with open(filename_archi, 'w') as f: # save model architecture
        f.write(model.to_json())
    return model

# # train undistilled model by executing the function
# train("models/MNIST/CNN_model.h5",
#       "models/MNIST/CNN_weights.h5",
#       "models/MNIST/CNN_architecture.json",
#       params,
#       epochs=1,
#       batch_size=128,
#       train_temp=1)

# calculate accuracy of the models
# model loading/reconstruction from JSON file  
with open('models/MNIST/CNN_architecture.json', 'r') as f:
    model = model_from_json(f.read())  
# load weights into the new model
model.load_weights('models/MNIST/CNN_weights.h5')

# function to get model accuracy scores
def eval_model(model):  
    global wrong_predictions_num, corr_predictions_num,total_pred
    wrong_predictions_num = 0
    corr_predictions_num = 0

    for n in range(x_test.shape[0]):
        # get true label values
        label_true = y_test[n].argmax()

        # get_prediction
        predictions = model.predict(np.reshape(x_test[n],[1,28,28,1]))
        predictions_lst = predictions.tolist()[0]
        max_value = max(predictions_lst)
        predicted = predictions_lst.index(max_value)

        # compute percentages
        if label_true != predicted:
            wrong_predictions_num += 1
        elif label_true == label_true:
            corr_predictions_num += 1
    
    # print results
    total_pred = corr_predictions_num+wrong_predictions_num # total number of predictions
    print('number of correct predictions: ', corr_predictions_num)
    print('number of incorrect predictions: ', wrong_predictions_num)
    print('total: ', total_pred)
    print('Accuracy: ', round(corr_predictions_num/total_pred*100,2),'%')
    
# get model accuracy scores 
print('Model Accuracy MNIST \n')
eval_model(model)

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

count = 0
for i in range(0,y_test.shape[0]-1):
	try:
		store_images_fgsm(i)
	except:
		pass
	count +=1
	print('iteration:', count)
