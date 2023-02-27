
from data import mnist_data
from LeNet5_model import LeNet5
import matplotlib.pyplot as plt

batch_size = 64
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
log_csv = CSVLogger('./lenet5_logs.csv', separator=',', append=False)
callbacks_list = [early_stop, log_csv]

X_train,Y_train,X_test,Y_test=mnist_data()
input_shape=X_train[0,:,:,:].shape
n_classes=10

model = LeNet5(input_shape, n_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, validation_split=0.2,callbacks=callbacks_list)
model.save('./LeNet5.hdf5')

score = model.evaluate(X_test, Y_test)
print('Test Loss= ', score)

import random
import numpy as np
num=random.randint(0,len(Y_test)-1)
plt.imshow(X_test[num])
print('Original Label:',np.argmax(Y_test[num]))
print('Predicted Label:',np.argmax(predictions[num]))
