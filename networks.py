import numpy as np
import keras
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import AveragePooling2D,GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Conv2DTranspose
from keras.layers.merge import concatenate #Concatenate (capital C) not working 

# functions implementing different cnn architectures

def UNET(input): #simple UNet

	#model parameters
	bnorm_axis = -1
	#filter sizes of the original model
	nfilters = np.array([64, 128, 256, 512, 1024])

	#downsize the UNET for this example.
	#the smaller network is faster to train
	#and produces excellent results on the dataset at hand
	nfilters = (nfilters/8).astype('int')

	#input
	input_tensor = Input(shape=input.shape[1:], name='input_tensor')

	####################################
	# encoder (contracting path)
	####################################
	#encoder block 0
	e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(input_tensor)
	e0 = BatchNormalization(axis=bnorm_axis)(e0)
	e0 = Activation('relu')(e0)
	e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(e0)
	e0 = BatchNormalization(axis=bnorm_axis)(e0)
	e0 = Activation('relu')(e0)

	#encoder block 1
	e1 = MaxPooling2D((2, 2))(e0)
	e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
	e1 = BatchNormalization(axis=bnorm_axis)(e1)
	e1 = Activation('relu')(e1)
	e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
	e1 = BatchNormalization(axis=bnorm_axis)(e1)
	e1 = Activation('relu')(e1)

	#encoder block 2
	e2 = MaxPooling2D((2, 2))(e1)
	e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
	e2 = BatchNormalization(axis=bnorm_axis)(e2)
	e2 = Activation('relu')(e2)
	e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
	e2 = BatchNormalization(axis=bnorm_axis)(e2)
	e2 = Activation('relu')(e2)

	#encoder block 3
	e3 = MaxPooling2D((2, 2))(e2)
	e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
	e3 = BatchNormalization(axis=bnorm_axis)(e3)
	e3 = Activation('relu')(e3)
	e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
	e3 = BatchNormalization(axis=bnorm_axis)(e3)
	e3 = Activation('relu')(e3)

	#encoder block 4
	e4 = MaxPooling2D((2, 2))(e3)
	e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
	e4 = BatchNormalization(axis=bnorm_axis)(e4)
	e4 = Activation('relu')(e4)
	e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
	e4 = BatchNormalization(axis=bnorm_axis)(e4)
	e4 = Activation('relu')(e4)
	#e4 = MaxPooling2D((2, 2))(e4)
	####################################
	# decoder (expansive path)
	####################################

	#decoder block 3
	d3=UpSampling2D((2, 2),)(e4)
	d3=concatenate([e3,d3], axis=-1)#skip connection
	d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
	d3=BatchNormalization(axis=bnorm_axis)(d3)
	d3=Activation('relu')(d3)
	d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
	d3=BatchNormalization(axis=bnorm_axis)(d3)
	d3=Activation('relu')(d3)

	#decoder block 2
	d2=UpSampling2D((2, 2),)(d3)
	d2=concatenate([e2,d2], axis=-1)#skip connection
	d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
	d2=BatchNormalization(axis=bnorm_axis)(d2)
	d2=Activation('relu')(d2)
	d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
	d2=BatchNormalization(axis=bnorm_axis)(d2)
	d2=Activation('relu')(d2)

	#decoder block 1
	d1=UpSampling2D((2, 2),)(d2)
	d1=concatenate([e1,d1], axis=-1)#skip connection
	d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
	d1=BatchNormalization(axis=bnorm_axis)(d1)
	d1=Activation('relu')(d1)
	d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
	d1=BatchNormalization(axis=bnorm_axis)(d1)
	d1=Activation('relu')(d1)

	#decoder block 0
	d0=UpSampling2D((2, 2),)(d1)
	d0=concatenate([e0,d0], axis=-1)#skip connection
	d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
	d0=BatchNormalization(axis=bnorm_axis)(d0)
	d0=Activation('relu')(d0)
	d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
	d0=BatchNormalization(axis=bnorm_axis)(d0)
	d0=Activation('relu')(d0)

	#output
	out_class = Dense(1)(d0)
	out_class = Activation('sigmoid',name='output')(out_class)

	#create and compile the model
	model=Model(inputs=input_tensor,outputs=out_class)
	model.compile(loss={'output':'binary_crossentropy'},
				  metrics={'output':'accuracy'},
				  optimizer='adam')
	#model.summary()
	#plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=True)
	return model


def UNET_mc(input): #UNET with dropout layers (p=0.5)

	#model parameters
	bnorm_axis = -1
	#filter sizes of the original model
	nfilters = np.array([64, 128, 256, 512, 1024])

	#downsize the UNET for this example.
	#the smaller network is faster to train
	#and produces excellent results on the dataset at hand
	nfilters = (nfilters/8).astype('int')

	#input
	input_tensor = Input(shape=input.shape[1:], name='input_tensor')

	####################################
	# encoder (contracting path)
	####################################
	#encoder block 0
	e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(input_tensor)
	e0 = BatchNormalization(axis=bnorm_axis)(e0)
	e0 = Activation('relu')(e0)
	e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(e0)
	e0 = BatchNormalization(axis=bnorm_axis)(e0)
	e0 = Activation('relu')(e0)

	#encoder block 1
	e1 = MaxPooling2D((2, 2))(e0)
	e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
	e1 = BatchNormalization(axis=bnorm_axis)(e1)
	e1 = Activation('relu')(e1)
	e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
	e1 = BatchNormalization(axis=bnorm_axis)(e1)
	e1 = Activation('relu')(e1)

	#encoder block 2
	e2 = MaxPooling2D((2, 2))(e1)
	e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
	e2 = BatchNormalization(axis=bnorm_axis)(e2)
	e2 = Activation('relu')(e2)
	e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
	e2 = BatchNormalization(axis=bnorm_axis)(e2)
	e2 = Activation('relu')(e2)
	e2 = Dropout(0.5)(e2,training=True)

	#encoder block 3
	e3 = MaxPooling2D((2, 2))(e2)
	e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
	e3 = BatchNormalization(axis=bnorm_axis)(e3)
	e3 = Activation('relu')(e3)
	e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
	e3 = BatchNormalization(axis=bnorm_axis)(e3)
	e3 = Activation('relu')(e3)
	e3 = Dropout(0.5)(e3,training=True)

	#encoder block 4
	e4 = MaxPooling2D((2, 2))(e3)
	e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
	e4 = BatchNormalization(axis=bnorm_axis)(e4)
	e4 = Activation('relu')(e4)
	e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
	e4 = BatchNormalization(axis=bnorm_axis)(e4)
	e4 = Activation('relu')(e4)
	#e4 = MaxPooling2D((2, 2))(e4)
	e4 = Dropout(0.5)(e4,training=True)
    ####################################
	# decoder (expansive path)
	####################################

	#decoder block 3
	d3=UpSampling2D((2, 2),)(e4)
	d3=concatenate([e3,d3], axis=-1)#skip connection
	d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
	d3=BatchNormalization(axis=bnorm_axis)(d3)
	d3=Activation('relu')(d3)
	d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
	d3=BatchNormalization(axis=bnorm_axis)(d3)
	d3=Activation('relu')(d3)
	d3 = Dropout(0.5)(d3,training=True)

	#decoder block 2
	d2=UpSampling2D((2, 2),)(d3)
	d2=concatenate([e2,d2], axis=-1)#skip connection
	d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
	d2=BatchNormalization(axis=bnorm_axis)(d2)
	d2=Activation('relu')(d2)
	d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
	d2=BatchNormalization(axis=bnorm_axis)(d2)
	d2=Activation('relu')(d2)
	d2 = Dropout(0.5)(d2,training=True)

	#decoder block 1
	d1=UpSampling2D((2, 2),)(d2)
	d1=concatenate([e1,d1], axis=-1)#skip connection
	d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
	d1=BatchNormalization(axis=bnorm_axis)(d1)
	d1=Activation('relu')(d1)
	d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
	d1=BatchNormalization(axis=bnorm_axis)(d1)
	d1=Activation('relu')(d1)
	d1 = Dropout(0.5)(d1,training=True)

	#decoder block 0
	d0=UpSampling2D((2, 2),)(d1)
	d0=concatenate([e0,d0], axis=-1)#skip connection
	d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
	d0=BatchNormalization(axis=bnorm_axis)(d0)
	d0=Activation('relu')(d0)
	d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
	d0=BatchNormalization(axis=bnorm_axis)(d0)
	d0=Activation('relu')(d0)

	#output
	out_class = Dense(1)(d0)
	out_class = Activation('sigmoid',name='output')(out_class)

	#create and compile the model
	model=Model(inputs=input_tensor,outputs=out_class)
	model.compile(loss={'output':'binary_crossentropy'},
	              metrics={'output':'accuracy'},
	              optimizer='adam')
	#plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=True)
	return model



def DiceNet(input1):
	
	#creates vgg-style cnn for segmentation quality estimation

	#model parameters
	bnorm_axis = -1
	#filter sizes of the original model
	nfilters = np.array([32, 64, 128, 256])

	#inputs
	input_tensor = Input(shape=input1.shape[1:], name='input')
	
		#Conv block #1
	x = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(input_tensor)
	x = BatchNormalization(axis=bnorm_axis)(x)
	x = Activation('relu')(x)
	x = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization(axis=bnorm_axis)(x)
	x = Activation('relu')(x)

	#max-pooling #1 
	x = MaxPooling2D((2, 2))(x)

	#Conv block #2
	x = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization(axis=bnorm_axis)(x)
	x = Activation('relu')(x)
	x = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization(axis=bnorm_axis)(x)
	x = Activation('relu')(x)

	#max-pooling #2
	x = MaxPooling2D((2, 2))(x)

	#Conv block #3
	x = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization(axis=bnorm_axis)(x)
	x = Activation('relu')(x)
	x = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization(axis=bnorm_axis)(x)
	x = Activation('relu')(x)

	#max-pooling #3
	x = MaxPooling2D((2, 2))(x)

	#Conv block #4
	x = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization(axis=bnorm_axis)(x)
	x = Activation('relu')(x)
	x = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(x)
	x = BatchNormalization(axis=bnorm_axis)(x)
	x = Activation('relu')(x)

	#global average pooling
	x = GlobalAveragePooling2D()(x)

	#output
	x = Dense(1)(x)
	output = Activation('sigmoid',name='output')(x)

	#create and compile the model
	model=Model(inputs=input_tensor,outputs=output)
	model.compile(loss={'output':'mse'}, optimizer='adam')

	return model