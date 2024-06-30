"""
Custom losses.
"""
import tensorflow as tf

def mae(y_true,y_pred):
    return 0

def SSIMLoss(y_true,y_pred):
    loss = 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,300.0)) # max_val is the dynamic range of the image.
    return loss