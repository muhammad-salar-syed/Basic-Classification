

from keras import datasets
import keras

def mnist_data():
    n_classes = 10

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    img_rows, img_cols = x_train.shape[1:]


    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    X_train = x_train/255
    X_test = x_test/255

    Y_train = keras.utils.to_categorical(y_train, n_classes)
    Y_test = keras.utils.to_categorical(y_test, n_classes)

    return X_train,Y_train,X_test,Y_test
        
