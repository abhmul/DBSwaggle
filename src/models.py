import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

def conv3D(img_dim = (50, 50, 20), nb_filters = (32, 64), conv_size = 3, pool_size = 2):
    img_rows, img_cols, img_depth = img_dim[0], img_dim[1], img_dim[2]
    nb_conv = [conv_size, conv_size, conv_size]
    nb_pool = [pool_size, pool_size, pool_size]
    nb_classes = 2

    model = Sequential()

    # Add first layer
    model.add(Convolution3D(nb_filters[0],nb_depth=nb_conv[0], nb_row=nb_conv[0], nb_col=nb_conv[0], input_shape=(img_rows, img_cols, 1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

    # Add another layer
    model.add(Convolution3D(nb_filters[1],nb_depth=nb_conv[0], nb_row=nb_conv[0], nb_col=nb_conv[0], activation='relu'))
    model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))

    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='RMSprop')
    return model
