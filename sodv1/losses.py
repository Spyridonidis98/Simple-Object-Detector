import keras.backend as K
import tensorflow as tf
import numpy as np

def my_loss_v1(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def my_loss_v2(y_true, y_pred):
    conf = tf.square(y_true[...,0] - y_pred[...,0])
    box = tf.reduce_sum(tf.square(y_true[...,1:5] - y_pred[...,1:5]), axis=-1)
    classes = tf.reduce_sum(tf.square(y_true[...,5:]- y_pred[...,5:]), axis=-1)
    return tf.reduce_sum(conf+box+classes)/(y_true.shape[0]*y_true.shape[1]*y_true.shape[2]*y_true.shape[3])

def my_loss_v3(y_true, y_pred):
    iobj = y_true[..., 0]
    conf = tf.square(y_true[...,0] - y_pred[...,0])
    box = tf.multiply(tf.reduce_sum(tf.square(y_true[...,1:5] - y_pred[...,1:5]), axis=-1), iobj)
    classes = tf.multiply(tf.reduce_sum(tf.square(y_true[...,5:]- y_pred[...,5:]), axis=-1), iobj)
    return tf.reduce_sum(conf+box+classes)










