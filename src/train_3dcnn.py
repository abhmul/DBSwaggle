from keras.layers import Convolution3D
from keras.callbacks import ModelCheckpoint
import DataScienceBowl as dsb

import models

num_train_pics = 1000 # This number needs to be calculated
num_val_pics = 100 # This number needs to be calculated
nb_epoch = 25
best_model_file = '../models/conv3d2layer_weights.h5'
model_func = models.conv3d

# Instantiate the generators
train_gen = dsb.load_train(new_spacing=[1,1,1], threshold=-320, fill_lung_structures=True, norm=None, center_mean=None)
val_gen = dsb.load_val(new_spacing=[1,1,1], threshold=-320, fill_lung_structures=True, norm=None, center_mean=None)

# Instantiate the model
model = model_func()

# Instantiate the best model callback
# This will save the best scoring model weights to the parent directory
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', mode='min',
                             verbose=1, save_best_only=True,
                             save_weights_only=True)


model.fit_generator(train_gen, num_train_pics, verbose=1, callbacks=[best_model],
                    validation_data=val_gen, nb_val_samples=num_val_pics)
