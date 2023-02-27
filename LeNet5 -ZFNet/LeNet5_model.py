
from keras.models import Model
from keras.layers import Input,Dense,Flatten, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation,Conv2DTranspose, BatchNormalization,Dropout, Lambda
from tensorflow.keras.optimizers import Adam

def LeNet5(input_size,n_classes):
	inputs=Input(input_size,name='Input')
	
	C1=Conv2D(6,5,padding='same',kernel_initializer='he_normal',name='Conv1')(inputs)
	C1=BatchNormalization(name='BN_Conv1')(C1)
	C1=Activation('relu',name='Conv1_Activation')(C1)

	P1=MaxPooling2D(pool_size=(2,2),name='Conv1_Pooling')(C1)

	C2=Conv2D(16,5,padding='same',kernel_initializer='he_normal',name='Conv2')(P1)
	C2=BatchNormalization(name='BN_Conv2')(C2)
	C2=Activation('relu',name='Conv2_Activation')(C2)

	P2=MaxPooling2D(pool_size=(2,2),name='Conv2_Pooling')(C2)

	C3=Conv2D(120,5,padding='same',kernel_initializer='he_normal',name='Conv3')(P2)
	C3=BatchNormalization(name='BN_Conv3')(C3)
	C3=Activation('relu',name='Conv3_Activation')(C3)

	F1=Flatten()(C3)
	D1=Dense(84,name='Dense1')(F1)
	D1=Activation('relu',name='Dense1_Activation')(D1)

	D2=Dense(n_classes,name='Final')(D1)
	D2=Activation('softmax',name='Final_Activation')(D2)
	
	model=Model(inputs,D2,name='LeNet5')
	return model
