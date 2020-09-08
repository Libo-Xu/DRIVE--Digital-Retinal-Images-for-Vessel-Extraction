import os
import h5py
import numpy as np
from PIL import Image
    
# Train datasets paths
train_origi_path = './DRIVE/training/images/'
train_truth_path = './DRIVE/training/1st_manual/'
train_mask_path = './DRIVE/training/mask/'


# Test datasets paths
test_origi_path = './DRIVE/test/images/'
test_truth_path = './DRIVE/test/1st_manual/'
test_mask_path = './DRIVE/test/mask/'

N = 20
channels = 3
height = 584
width = 565

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)
        
def read_datasets(imgs_dir,truth_dir,mask_dir,train = True):
    imgs = np.empty((N,height,width,channels))
    truth = np.empty((N,height,width))
    masks = np.empty((N,height,width))
    for root, dirs, files in os.walk(imgs_dir):
        for i in range(N):
            # Original images
            img = Image.open(imgs_dir + files[i])
            imgs[i] = np.asarray(img)
            # Ground truth
            true = Image.open(truth_dir + files[i][0:2] + '_manual1.gif')
            truth[i] = np.asarray(true)
            # Masks
            if train:
                mask = Image.open(mask_dir + files[i][0:2] + '_training_mask.gif')
            else:
                mask = Image.open(mask_dir + files[i][0:2] + '_test_mask.gif')
            masks[i] = np.asarray(mask)
    # Reshaping the images 
    imgs = np.transpose(imgs,(0,3,1,2))
    truth = np.reshape(truth,(N,1,height,width))
    masks = np.reshape(masks,(N,1,height,width))
    
    return imgs, truth, masks

# Create new directory for datasets
path = './prepared_datasets/'
os.makedirs(path)
    
# Preparing the training datasets
imgs_train, truth_train, masks_train = read_datasets(train_origi_path, train_truth_path, train_mask_path, True)
write_hdf5(imgs_train, path + 'imgs_train.hdf5')
write_hdf5(truth_train, path + 'truth_train.hdf5')
write_hdf5(masks_train, path + 'masks_train.hdf5')
print('Train data done.')

# Preparing the testing datasets
imgs_test, truth_test, masks_test = read_datasets(test_origi_path, test_truth_path, test_mask_path, False)
write_hdf5(imgs_test, path + 'imgs_test.hdf5')
write_hdf5(truth_test, path + 'truth_test.hdf5')
write_hdf5(masks_test, path + 'masks_test.hdf5')
print('Test data done.')