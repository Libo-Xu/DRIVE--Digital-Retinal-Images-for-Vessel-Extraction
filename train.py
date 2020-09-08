import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from preprocessing import *
from Unet import get_UNet
from plot import plot_learning_curve

# Make sure the program will be running by GPU
import tensorflow as tf
tf.test.gpu_device_name()

# Set the path and parameters of getting patches 
N_patches = 100000
patch_h = 48
patch_w = 48
path_data = './prepared_datasets/'
imgs_train = path_data + 'imgs_train.hdf5'
truth_train = path_data + 'truth_train.hdf5'

# Get the patches of training images
patches_imgs_train, patches_masks_train = prepare_training_data(imgs_train, truth_train, patch_h, patch_w, N_patches)

# Parameters for training model
N_epochs = 150
batch_size = 64
lr = 0.1
decay_rate = lr / N_epochs
sgd = SGD(lr=lr, momentum=0.8, decay=decay_rate, nesterov=False)

model = get_UNet(img_shape=(patch_h,patch_w,1), Base=32, depth=4, inc_rate=2, 
                 activation='relu', drop=0.2, batchnorm=True, N=2)

model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

# Save the architecture of the model
json_string = model.to_json()
open('model_architecture.json', 'w').write(json_string)

# Save the best weights 
checkpointer = ModelCheckpoint(filepath= 'best_weights.h5', 
                               monitor='val_loss', 
                               verbose=1, 
                               save_best_only=True, 
                               save_weights_only=False, 
                               mode='auto') 
# Set early stopping 
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1,mode='min')

# 90% for traning and 10% for validation
History = model.fit(patches_imgs_train, 
                    patches_masks_train, 
                    epochs=N_epochs, 
                    batch_size=batch_size, 
                    verbose=2, 
                    shuffle=True, 
                    validation_split=0.1,
                    callbacks=[checkpointer,early_stopping])

# Plot the learning curve
plot_learning_curve(History, save_path + 'learning_curve')