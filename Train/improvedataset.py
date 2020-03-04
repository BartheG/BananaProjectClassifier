from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os

#Data augmentation -> Transforms same image to get new training data

def SizefindInDir(pathdir,ext):
	return [ os.path.join(root, file)\
		for root, dirs, files in os.walk(pathdir)\
			for file in files\
				if file.endswith(ext) ]

def improve(path,exportpath,prefix):
	datagen = ImageDataGenerator(
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest'
	)

	for name in path:
		img = load_img(name)
		x = img_to_array(img)
		x = x.reshape((1,) + x.shape)
		i = 0
		for batch in datagen.flow(x, batch_size=1,save_to_dir=exportpath, save_prefix=prefix, save_format='jpg'):
			i += 1
			if i > 20:
				break

#Validation
print('Validation good')
improve( SizefindInDir('../OriginalData/validation/ori_good','.jpg'),'validation/good','good' )
print('Validation flecked')
improve( SizefindInDir('../OriginalData/validation/ori_flecked','.jpg'),'validation/flecked','flecked' )

#Test
print('Train good')
improve( SizefindInDir('../OriginalData/train/ori_good','.jpg'),'train/good','good' )
print('Train flecked')
improve( SizefindInDir('../OriginalData/train/ori_flecked','.jpg'),'train/flecked','flecked' )

