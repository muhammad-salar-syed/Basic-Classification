
from keras.models import Model
from keras.layers import Input,Dense,Flatten, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation,Conv2DTranspose, BatchNormalization,Dropout, Lambda
from tensorflow.keras.optimizers import Adam

def ZFNet(input_size,n_classes):
    inputs=Input(input_size)

    C1=Conv2D(filters=96, kernel_size=(7,7), strides=(2,2), padding='valid')(inputs)
    C1=Activation('relu')(C1)
    P1=MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(C1)
    
    C2=Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding='valid')(P1)
    C2=Activation('relu')(C2)
    P2=MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(C2)
    
    C3=Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')(P2)
    C3=Activation('relu')(C3)

    C4=Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')(C3)
    C4=Activation('relu')(C4)
    
    C5=Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same')(C4)
    C5=Activation('relu')(C5)
    P3=MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(C5)
    
    F1=Flatten()(P3)
 
    D1=Dense(4096)(F1)
    D1=Activation('relu')(D1)
    
    D2=Dense(4096)(D1)
    D2=Activation('relu')(D2)
    
    D3=Dense(n_classes)(D2)
    D3=Activation('softmax')(D3)
    
    model=Model(inputs,D3,name='ZFNet')
    return model

