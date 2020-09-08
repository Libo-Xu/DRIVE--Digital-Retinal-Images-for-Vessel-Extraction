import os
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(History, fig_name):
    #%matplotlib inline
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend();
    plt.savefig(fig_name + '.png')
    plt.close()
    
def plot_validation_metric(History, fig_name):
    #%matplotlib inline
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve_metrics")
    plt.plot(History.history["val_dice_coef"], label="val_dice_coef")
    plt.plot(History.history["val_precision"], label="val_precision")
    plt.plot(History.history["val_recall"], label="val_recall")
    
    plt.xlabel("Epochs")
    plt.legend();
    plt.savefig(fig_name + '.png')
    plt.close()

