import numpy as np
import os

#Keras imports
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout
from keras import applications
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

#Base 64 Treatment
from io import BytesIO
import base64
import re
from PIL import Image

import tensorflow as tf

#Rebuild base model of our classifier et returns it.
def m_model():
	model = Sequential()
	model.add(Flatten(input_shape=(4,4,512)))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(optimizer='rmsprop',
		loss='binary_crossentropy', metrics=['accuracy'])
	return model

#Searchs recursively in 'pathdir' and returns every files ending by the extension specified in 'ext'
def findInDir(pathdir,ext):
	return [ os.path.join(root, file) for root, dirs, files in os.walk(pathdir) for file in files if file.endswith(ext) ]

#Wrapper to get classification with our model.
#The class will load the weights of our model and will be able to make predictions,
# on base64 picture passed throught parameter.
class GetModelPrediction:
	def __init__(self,w_path,model):
		self.model = model
		self.model.load_weights(w_path)
		self.b_model = applications.VGG16(include_top=False, weights='imagenet')
		self.graph = tf.get_default_graph()

	def process(self,path,is_base=False,m_shape=(150,150)):
		if is_base:
			image_data = re.sub('^data:image/.+;base64,', '', path)
			t_img = Image.open(
				BytesIO(base64.b64decode(image_data)))
			t_img = t_img.resize(m_shape, Image.ANTIALIAS)
		else:
			t_img = image.load_img(
				path,
				target_size=m_shape
			)
		t_x = image.img_to_array(t_img)
		t_x = np.expand_dims(t_x,axis=0)
		t_x = preprocess_input(t_x)
		return t_x

	def predict(self,topredict):
		with self.graph.as_default():
			features = self.b_model.predict( topredict/255.0 )
			return self.model.predict_classes(features)

	def run(self,img_path,is_base):
		if type(img_path)==list:
			return [ self.predict( self.process(i,is_base) ) for i in img_path ]
		else:
			return [self.predict( self.process(img_path,is_base) ) ]

#Initialize the Wrapper with the weigths of our model.
def m_predict():
	return GetModelPrediction(
		'./ExportModel/bottleneck_fc_model.h5',
		m_model()).run