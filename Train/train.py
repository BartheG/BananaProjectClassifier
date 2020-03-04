import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

import sys,os

def SizefindInDir(pathdir,ext):
	return len([ os.path.join(root, file)\
		for root, dirs, files in os.walk(pathdir)\
			for file in files\
				if file.endswith(ext) ])

class TrainingWrap:
	def __init__(self):
		if len(sys.argv)!=3:
			print('Error: Argument number')
			raise ValueError
		if not os.path.exists(sys.argv[1]) or not os.path.exists(sys.argv[2]):
			print('Error: Argument path not exists')
			raise FileNotFoundError
		self.args()

	def args(self,):
		self.img_width, self.img_height = 150, 150
		self.top_model_weights_path = './ExportTemp/bottleneck_fc_model.h5'
		self.train_data_dir = sys.argv[1]
		self.validation_data_dir = sys.argv[2]
		#self.nb_train_samples = SizefindInDir(self.train_data_dir,'.jpg')
		self.nb_train_samples = 4848
		self.nb_validation_samples = 1488
		#self.nb_validation_samples = SizefindInDir(self.validation_data_dir,'.jpg')
		self.epochs = 50
		self.batch_size = 16

		print('Train',self.train_data_dir,self.nb_train_samples)
		print('Test',self.validation_data_dir,self.nb_validation_samples)
		print('Size pic',self.img_width,self.img_height)
		print('Epochs',self.epochs,'Batch size',self.batch_size)

	def save_features(self,):
		datagen = ImageDataGenerator(rescale=1. / 255)

		# build the VGG16 network
		model = applications.VGG16(include_top=False, weights='imagenet')

		generator = datagen.flow_from_directory(
			self.train_data_dir,
			target_size=(self.img_width, self.img_height),
			batch_size=self.batch_size,
			class_mode=None,
			shuffle=False)
		print('Predict generator -> Saving features train in ExportTemp')
		bottleneck_features_train = model.predict_generator(
			generator, self.nb_train_samples // self.batch_size)
		np.save(open('./ExportTemp/bottleneck_features_train.npy', 'wb'),
			bottleneck_features_train)

		generator = datagen.flow_from_directory(
			self.validation_data_dir,
			target_size=(self.img_width, self.img_height),
			batch_size=self.batch_size,
			class_mode=None,
			shuffle=False)
		print('Validation generator -> Saving features validation in ExportTemp')
		bottleneck_features_validation = model.predict_generator(
			generator, self.nb_validation_samples // self.batch_size)
		np.save(open('./ExportTemp/bottleneck_features_validation.npy', 'wb'),
			bottleneck_features_validation)

	def top_model(self,):
		train_data = np.load(open('./ExportTemp/bottleneck_features_train.npy','rb'))

		train_labels = np.array([0] * int(self.nb_train_samples / 2) + [1] * int(self.nb_train_samples / 2))

		validation_data = np.load(open('./ExportTemp/bottleneck_features_validation.npy','rb'))
		validation_labels = np.array([0] * int(self.nb_validation_samples / 2) + [1] * int(self.nb_validation_samples / 2))

		model = Sequential()
		model.add(Flatten(input_shape=train_data.shape[1:]))
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1, activation='sigmoid'))

		model.compile(optimizer='rmsprop',
			loss='binary_crossentropy', metrics=['accuracy'])

		model.fit(train_data, train_labels,
			epochs=self.epochs,
			batch_size=self.batch_size,
			validation_data=(validation_data, validation_labels))
		top_model_weights_path = './ExportTemp/bottleneck_fc_model.h5'
		model.save_weights(top_model_weights_path)

def main():
	tw = TrainingWrap()
	tw.save_features()
	tw.top_model()

if __name__ == "__main__":
	main()
