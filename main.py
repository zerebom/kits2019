import argparse
import keras.callbacks
from PIL import Image, ImageOps
# from IPython.display import display_png
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from Models.Unet import UNet
import time

from tensorflow.python.keras.layers import Input
import tensorflow as tf
from datetime import datetime as dt
from tensorflow import keras
from Utils.reporter import Reporter
from Utils.loader import Loader
from Utils.status import ON_WIN

import json
import glob
from keras.layers.core import Lambda


SAVE_BATCH_SIZE = 2
WORKERS = 2
gpu_count=2

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

        y_trues = K.get_value(y_true_res[:,:,:,1])
        y_preds = K.get_value(y_pred_res[:, :, :, 1])

        return self.dice_coef(y_trues, y_preds)

    def dice_2(self, y_true, y_pred):
        y_true_res = tf.reshape(y_true, (-1, self.__input_h, self.__input_w, self.__class_num))
        y_pred_res = tf.reshape(y_pred, (-1, self.__input_h, self.__input_w, self.__class_num))

        y_trues = K.get_value(y_true_res[:,:,:, 2])
        y_preds = K.get_value(y_pred_res[:,:,:, 2])

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


def train(parser):
    config = json.load(open('./setting.json'))
    d_num = parser.d_num
    im_size = config['FIXED_SIZES'][d_num]

    START_TIME = time.time()
    reporter = Reporter(parser=parser)
    loader = Loader(parser=parser)

    train_gen, valid_gen, test_gen = loader.return_gen()
    train_steps, valid_steps, test_steps = loader.return_step()
    # ---------------------------model----------------------------------

    input_channel_count = parser.input_channel
    output_channel_count = 3
    first_layer_filter_count = parser.filter

    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count, im_size=im_size, parser=parser)

    model = network.get_model()
    # model.compile(loss=dice_coef_loss,optimizer='adam', metrics=[dice])
    optimizer = tf.keras.optimizers.Adam(lr=parser.trainrate)

    model.compile(
        loss=[DiceLossByClass(im_size, 3).dice_coef_loss], optimizer=optimizer, metrics=[dice,dice_1,dice_2])

    model.summary()
    # ---------------------------training----------------------------------

    batch_size = parser.batch_size
    epochs = parser.epoch

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    logdir = os.path.join('./logs', dt.today().strftime("%Y%m%d_%H%M"))
    os.makedirs(logdir, exist_ok=True)

    tb_cb = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True)
    es_cb = EarlyStopping(monitor='val_loss', patience=parser.early_stopping, verbose=1, mode='auto')

    print("start training.")
    # Pythonジェネレータ（またはSequenceのインスタンス）によりバッチ毎に生成されたデータでモデルを訓練します．
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        max_queue_size=16,
        workers= 1 if ON_WIN else WORKERS*gpu_count,
        use_multiprocessing=False if ON_WIN else True,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        # use_multiprocessing=True,
        callbacks=[es_cb, tb_cb])

    print("finish training. And start making predict.")

    # ---------------------------predict----------------------------------

    test_preds = model.predict_generator(test_gen, steps=test_steps, verbose=1)

    ELAPSED_TIME = int(time.time() - START_TIME)
    reporter.add_log_documents(f'ELAPSED_TIME:{ELAPSED_TIME} [sec]')

    # ---------------------------report----------------------------------

    parser.save_logs = True
    if parser.save_logs:
        reporter.add_val_loss(history.history['val_loss'])
        reporter.add_val_dice(history.history['val_dice'])

        reporter.add_model_name(network.__class__.__name__)
        reporter.generate_main_dir()
        reporter.plot_history(history)
        reporter.save_params(history)

        train_gen, valid_gen, test_gen = loader.return_gen()

        for i in range(min(train_steps, SAVE_BATCH_SIZE)):
            batch_input, batch_teach = next(train_gen)
            batch_preds = model.predict(batch_input)
            reporter.plot_predict(batch_input, batch_preds, batch_teach, 'train', batch_num=i)

        for i in range(min(valid_steps, SAVE_BATCH_SIZE)):
            batch_input, batch_teach = next(valid_gen)
            batch_preds = model.predict(batch_input)
            reporter.plot_predict(batch_input, batch_preds, batch_teach, 'valid', batch_num=i)

        for i in range(min(test_steps, SAVE_BATCH_SIZE)):
            batch_input, batch_teach = next(test_gen)
            batch_preds = model.predict(batch_input)
            reporter.plot_predict(batch_input, batch_preds, batch_teach, 'test', batch_num=i)


def get_parser():
    parser = argparse.ArgumentParser(
        prog='generate parallax image using U-Net',
        usage='python main.py',
        description='This module　generate parallax image using U-Net.',
        add_help=True
    )

    parser.add_argument('-e', '--epoch', type=int,
                        default=200, help='Number of epochs')
    parser.add_argument('-f', '--filter', type=int,
                        default=48 if ON_WIN else 64, help='Number of model first_filters')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32 if ON_WIN else 64, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float,
                        default=1e-3, help='Training rate')
    parser.add_argument('-es', '--early_stopping', type=int,
                        default=10, help='early_stopping patience')
    parser.add_argument('-i', '--input_channel', type=int,
                        default=1, help='input_channel')
    parser.add_argument('-d', '--d_num', type=int,
                        default=1, help='directory_number')

    parser.add_argument('-a', '--augmentation',
                        action='store_true', help='Number of epochs')
    parser.add_argument('-s', '--save_logs',
                        action='store_true', help='save or not logs')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
