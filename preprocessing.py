import numpy as np
from PIL import Image
import cv2
import random
import h5py

        
def load_hdf5(infile):
    with h5py.File(infile,"r") as f:  
        return f["image"][()]  
    
##############################
####### Pre-processing #######
##############################
def data_preprocess(imgs):
    imgs = rgb2gray(imgs)
    imgs = normalization(imgs)
    imgs = clahe_equalized(imgs)
    imgs = adjust_gamma(imgs, 1.2)
    imgs = imgs/255.  
    return imgs

# Convert RGB image to grayscale 
def rgb2gray(rgb):
    gray = rgb[:,0,...] * 0.299 + rgb[:,1,...] * 0.587 + rgb[:,2,...] * 0.114
    gray = np.reshape(gray,(gray.shape[0],1,gray.shape[1],gray.shape[2]))
    return gray

# Z-score and min-max normalization of dataset
def normalization(imgs):
    #imgs_normalized = np.empty(imgs.shape)
    std = np.std(imgs)
    mean = np.mean(imgs)
    imgs_normalized = (imgs - mean) / std
     
    for i in range(imgs.shape[0]):
        imgs_max = np.max(imgs_normalized[i])
        imgs_min = np.min(imgs_normalized[i])
        imgs_normalized[i] = ((imgs_normalized[i] - imgs_min) / (imgs_max-np.min(imgs_normalized[i])))
        
    imgs_normalized = imgs_normalized * 255
    return imgs_normalized

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(imgs.shape[0]):
        imgs[i,0] = clahe.apply(imgs[i,0].astype(np.uint8))
    return imgs

# Gamma Correction
def adjust_gamma(imgs, gamma=1.0):
    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype(np.uint8)
    for i in range(imgs.shape[0]):
        imgs[i,0] = cv2.LUT(imgs[i,0].astype(np.uint8), table)
    return imgs


#################################
####### Data augmentation #######
#################################
# Extract patches randomly in training images
def extract_random(imgs, masks, patch_h, patch_w, N_patches): 
    N_imgs = imgs.shape[0]
    N_channels = imgs.shape[1]
    img_h = imgs.shape[2]  
    img_w = imgs.shape[3] 
    
    # Number of patches per image
    N_patch_img = int(N_patches / N_imgs)
    
    # Initialize the matrix for patches
    patches_imgs = np.empty((N_patches, N_channels, patch_h, patch_w))
    patches_masks = np.empty((N_patches, N_channels,  patch_h, patch_w))
    count = 0   
    for i in range(N_imgs):  
        N = 0
        while N < N_patch_img:
            # Get the coordinate of a center point randomly
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            # Extract a patch based on the center coordinate
            patch_img = imgs[i,:,(y_center-patch_h//2):(y_center+patch_h//2),(x_center-patch_w//2):(x_center+patch_w//2)]
            patch_mask = masks[i,:,(y_center-patch_h//2):(y_center+patch_h//2),(x_center-patch_w//2):(x_center+patch_w//2)]
            patches_imgs[count] = patch_img
            patches_masks[count]=patch_mask
            count +=1  
            N+=1  
    return patches_imgs, patches_masks

#Load the original data and return the extracted patches for training/testing
def prepare_training_data(imgs_train, truth_train, patch_h, patch_w, N_patches):
    train_imgs = load_hdf5(imgs_train)
    train_masks = load_hdf5(truth_train) #masks always the same
    # visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

    train_imgs = data_preprocess(train_imgs)
    train_masks = train_masks/255.
    
    # Cut the border at the top and the bottom to make images square
    train_imgs = train_imgs[:,:,9:574,:]
    train_masks = train_masks[:,:,9:574,:]  
    
    # Extract patches from traning images
    patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks, patch_h,patch_w,N_patches)
    
    # Reshape and make the masks one-hot to fit the output of U-net
    patches_imgs_train = np.reshape(patches_imgs_train,(N_patches,patch_h,patch_w,1))
    patches_masks_train = np.reshape(patches_masks_train,(N_patches,patch_h,patch_w,1))
    temp = np.zeros((N_patches,patch_h,patch_w,2))
    
    for i in range(N_patches):
        for j in range(patch_h):
            for k in range(patch_w):
                # Assign label [0,1] to vessel pixel and label [1,0] to background pixel
                if patches_masks_train[i,j,k,0] == 1:
                    temp[i,j,k,0] = 0
                    temp[i,j,k,1] = 1
                else:
                    temp[i,j,k,0] = 1
                    temp[i,j,k,1] = 0
    
    patches_masks_train = temp
    
    print('shape of train images patches: '+ str(patches_imgs_train.shape))
    print('shape of train masks patches: '+ str(patches_masks_train.shape))
    return patches_imgs_train, patches_masks_train

# Pad the images to fit the stride of extracting patches
def pad_overlap(imgs, patch_h, patch_w, stride_h, stride_w):
    N_imgs = imgs.shape[0]
    N_channels = imgs.shape[1]
    img_h = imgs.shape[2]  
    img_w = imgs.shape[3]
    pad_h = stride_h - (img_h-patch_h)%stride_h 
    pad_w = stride_w - (img_w-patch_w)%stride_w
    if ((img_h-patch_h)%stride_h != 0):  
        new_imgs = np.zeros((N_imgs, N_channels, img_h+pad_h, img_w))
        new_imgs[:,:,0:img_h,0:img_w] = imgs
        imgs = new_imgs
    if ((img_w-patch_w)%stride_w != 0):   
        new_imgs = np.zeros((N_imgs, N_channels, imgs.shape[2], img_w+pad_w))
        new_imgs[:,:,:,0:img_w] = imgs
        imgs = new_imgs
    # New shape would be [N_imgs, N_channels, img_h+pad_h, img_w+pad_w]
    print("images shape after padding: \n" +str(imgs.shape))
    return imgs

# Divide the images into overlapping patches in order (good for recomposing the patches into images)
def extract_ordered_overlap(imgs, patch_h, patch_w, stride_h, stride_w):
    N_imgs = imgs.shape[0]
    N_channels = imgs.shape[1]
    img_h = imgs.shape[2]  
    img_w = imgs.shape[3]
    N_patch_h = (img_h-patch_h)//stride_h+1
    N_patch_w = (img_w-patch_w)//stride_w+1
    # Number of patches per image
    N_patch_img = N_patch_h * N_patch_w 
    # Number of patches in total
    N_patch_total = N_patch_img * N_imgs
    patches = np.empty((N_patch_total, N_channels, patch_h, patch_w))
    count = 0
    for i in range(N_imgs):
        for h in range(N_patch_h):
            for w in range(N_patch_w):
                patch = imgs[i, :, (h*stride_h):((h*stride_h)+patch_h), (w*stride_w):((w*stride_w)+patch_w)]
                patches[count] = patch
                count +=1   #total
    return patches  

# Extract patches from testing images
def prepare_testing_data(imgs_test, truth_test, patch_h,
                             patch_w, stride_h, stride_w):
    
    test_imgs = load_hdf5(imgs_test)
    test_truth = load_hdf5(truth_test)

    test_imgs = data_preprocess(test_imgs)
    test_truth = test_truth/255.
    
    # Pad the images to fit the stride of extracting patches
    test_imgs = pad_overlap(test_imgs, patch_h, patch_w, stride_h, stride_w)

    # Divide the images into overlapping patches in order
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_h, patch_w, 
                                                stride_h, stride_w)
    patches_imgs_test = np.reshape(patches_imgs_test, (patches_imgs_test.shape[0],patch_h,
                                                       patch_w,patches_imgs_test.shape[1]))
    print('shape of test images patches: '+ str(patches_imgs_test.shape))
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_truth

# Restore the patches to the full-size images
def restore_patches(pred, N_imgs, img_h, img_w, stride_h, stride_w):
    N_channels = pred.shape[1]
    patch_h = pred.shape[2]
    patch_w = pred.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    
    # Initilize the probability of being vessel pixel and the number of overlapping for each pixel
    prob = np.zeros((N_imgs,N_channels,img_h,img_w)) 
    N_overlap = np.zeros((N_imgs,N_channels,img_h,img_w))

    count = 0
    for i in range(N_imgs):
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                prob[i, :, (h*stride_h):((h*stride_h)+patch_h), 
                     (w*stride_w):((w*stride_w)+patch_w)] += pred[count]
                N_overlap[i, :, (h*stride_h):((h*stride_h)+patch_h), 
                          (w*stride_w):((w*stride_w)+patch_w)] += 1
                count += 1
                
    # Calculate the average of probabilities for each pixel 
    avg_pred = prob / N_overlap
    
    return avg_pred

# Remove the results outside the FOV and only evaluate by results inside the FOV 
def clean_outside(imgs,truth, masks):
    N_imgs = imgs.shape[0]
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]
    
    y_pred = []
    y_true = []

    for i in range(N_imgs):  
        for h in range(img_h):
            for w in range(img_w):
                if masks[i, :, h, w] != 0:
                    y_pred.append(imgs[i, :, h, w])
                    y_true.append(truth[i, :, h, w])
                    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return y_pred, y_true              



