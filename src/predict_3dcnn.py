import DataScienceBowl as dsb
import models

num_test_pics = 1000 # This number needs to be calculated
best_model_file = '../models/conv3d2layer_weights.h5'
model_func = models.asd

# Instantiate the generators
test_gen = dsb.load_test(new_spacing=[1,1,1], threshold=-320, fill_lung_structures=True, norm=None, center_mean=None)

# Instantiate the model
model = model_func()

# Get the predictions
predictions = model.predict_generator(test_gen, num_test_pics, verbose=1)

dsb.create_submission(predictions)
