import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from preprocessing import *
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

# Make sure the program will be running by GPU
import tensorflow as tf
tf.test.gpu_device_name()

####### Prepare the dataset and model #######
#############################################
# Set the path and parameters of getting patches 
patch_h = 48
patch_w = 48
stride_h = 5
stride_w = 5
path_data = './prepared_datasets/'
imgs_test = path_data + 'imgs_test.hdf5'
masks_test = path_data + 'masks_test.hdf5'
truth_test = path_data + 'truth_test.hdf5'

# Original images
test_imgs = data_preprocess(test_imgs)
test_imgs = load_hdf5(imgs_test)

# Masks
test_imgs = data_preprocess(test_imgs)
test_masks = load_hdf5(masks_test)

# Number of original images
N_imgs = test_imgs.shape[0]

# Height and width of original image

img_h = test_imgs.shape[2]
img_w = test_imgs.shape[3]

# Get the patches of testing images
patches_imgs_test, new_h, new_w, test_truth = prepare_testing_data(imgs_test, truth_test, patch_h, patch_w, stride_h, stride_w)

# Load the model and the best weight
with open(r'model_architecture.json', 'r') as file:
    model_json1 = file.read()
model = model_from_json(model_json1)
model.load_weights('best_weights.h5')


#######  Predict #######
########################

predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)

pred = np.empty((predictions.shape[0],1,patch_h,patch_w))
for i in range(predictions.shape[0]):
    for j in range(patch_h):
        for k in range(patch_w):
            # Take the probability of being vessel pixel as the predicted result for each pixel 
            pred[i,0,j,k] = predictions[i,j,k,1]

# Restore the patches to the full-size images
pred_imgs = restore_patches(pred, N_imgs, new_h, new_w, stride_h, stride_w)
pred_imgs = pred_imgs[:, :, 0:img_h, 0:img_w]

####### Evaluate the results #######
####################################
# Remove the results outside the FOV and only evaluate by results inside the FOV 
y_pred, y_true = clean_outside(pred_imgs, test_truth, test_masks)

# AUC --- Area under the ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred)
print("\nArea under the ROC curve: " +str(auc))
plt.figure()
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("ROC_curve.png")

# Confusion matrix and related metrics
y_true = (y_true).astype(np.int)
for i in range(y_pred.shape[0]):
    if y_pred[i] < 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1
cm = confusion_matrix(y_true, y_pred)
accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) # sensitivity
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
f1 = 2 * cm[1,1] / (2 * cm[1,1] + cm[1, 0] + cm[0, 1])

print(cm)
print("accuracy: {:.4f}".format(accuracy))
print("recall: {:.4f}".format(recall))
print("precision: {:.4f}".format(precision))
print("specificity: {:.4f}".format(specificity))
print("F1: {:.4f}".format(f1))

# Reshape the images for visualization
test_imgs = np.reshape(test_imgs ,(N_imgs, img_h, img_w))
test_truth = np.reshape(test_truth ,(N_imgs, img_h, img_w))
pred_imgs = np.reshape(pred_imgs ,(N_imgs, img_h, img_w))

# Compare the original images, ground truth images and the segmentation results images
for i in range(N_imgs):
    display_img = np.concatenate((test_imgs[i],test_truth[i],pred_imgs[i]),axis = 1)
    display_img = (display_img * 255).astype(np.uint8)
    display_img = Image.fromarray(display_img)
    display_img.save(str(i+1) + '.png')

