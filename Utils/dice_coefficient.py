
"""
import keras.backend as KB
import tensorflow as tf
def dice_coef(y_true, y_pred):
    y_true = KB.flatten(y_true)
    y_pred = KB.flatten(y_pred)
    intersection = KB.sum(y_true * y_pred)
    return (2.0 * intersection + 1) / (KB.sum(y_true) + KB.sum(y_pred) + 1)
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
"""

import keras.backend as KB
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

class DiceLossByClass():
    def __init__(self, input_shape, class_num):
        self.__input_h = input_shape
        self.__input_w = input_shape
        self.__class_num = class_num

    def dice_coef(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        denominator = K.sum(y_true) + K.sum(y_pred)
        if denominator == 0:
            return 1
        if intersection == 0:
            return 1 / (denominator + 1)
        return (2.0 * intersection) / denominator
        # return (2.0 * intersection + 1) / (K.sum(y_true) + K.sum(y_pred) + 1)

    def dice_1(self, y_true, y_pred):
        # (N, h, w, ch)
        y_true_res = tf.reshape(y_true, (-1, self.__input_h, self.__input_w, self.__class_num))
        y_pred_res = tf.reshape(y_pred, (-1, self.__input_h, self.__input_w, self.__class_num))

        y_trues = K.get_value(y_true_res[:, :, :, 1])
        y_preds = K.get_value(y_pred_res[:, :, :, 1])

        return self.dice_coef(y_trues, y_preds)

    def dice_2(self, y_true, y_pred):
        y_true_res = tf.reshape(y_true, (-1, self.__input_h, self.__input_w, self.__class_num))
        y_pred_res = tf.reshape(y_pred, (-1, self.__input_h, self.__input_w, self.__class_num))

        y_trues = K.get_value(y_true_res[:, :, :, 2])
        y_preds = K.get_value(y_pred_res[:, :, :, 2])

        return self.dice_coef(y_trues, y_preds)

    def dice_coef_loss(self, y_true, y_pred):
        # (N, h, w, ch)
        y_true_res = tf.reshape(y_true, (-1, self.__input_h, self.__input_w, self.__class_num))
        y_pred_res = tf.reshape(y_pred, (-1, self.__input_h, self.__input_w, self.__class_num))
        # チャンネルごとのリストになる？
        y_trues = tf.unstack(y_true_res, axis=3)
        y_preds = tf.unstack(y_pred_res, axis=3)

        losses = []
        for y_t, y_p in zip(y_trues, y_preds):
            losses.append((1 - self.dice_coef(y_t, y_p)) * 3)
        return tf.reduce_sum(tf.stack(losses))
        # return 1 - self.dice_coef(y_true, y_pred)


def dice(y_true, y_pred):
    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    # cast->型変換,minimum2つのテンソルの要素ごとの最小値,equal->boolでかえってくる
    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice = 2. * intersection / (union + eps)
    return dice


def dice_1(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)

    truelabels = K.cast(y_true[:, :, :, 1], tf.int32)
    predictions = K.cast(y_pred[:, :, :, 1], tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice_1 = 2 * intersection / (union + eps)

    return dice_1


def dice_2(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = K.cast(y_true[:, :, :, 2], tf.int32)
    predictions = K.cast(y_pred[:, :, :, 2], tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice_2 = 2 * intersection / (union + eps)

    return dice_2


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice(y_true, y_pred)


def penalty_categorical(y_true,y_pred):
    array_tf = tf.convert_to_tensor(y_true,dtype=tf.float32)
    pred_tf = tf.convert_to_tensor(y_pred,dtype=tf.float32)

    epsilon = K.epsilon()

    result = tf.reduce_sum(array_tf,[0,1,2,3])

    result_pow = tf.pow(result,1.0/3.0)
    weight_y = result_pow / tf.reduce_sum(result_pow)

    return (-1) * tf.reduce_sum( 1 / (weight_y + epsilon) * array_tf * tf.log(pred_tf + epsilon),axis=-1)
