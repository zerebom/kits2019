import argparse
import keras.callbacks
from PIL import Image, ImageOps
from IPython.display import display_png
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

import json
import glob



SAVE_BATCH_SIZE = 2

# def dice(y_true, y_pred):

#     eps = K.constant(1e-6)
#     truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
#     predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

#     intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
#     union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
#     dice = 2. * intersection / (union + eps)
#     return dice

def dice(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection+1) / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice(y_true, y_pred)

def train(parser):

    config = json.load(open('./setting.json'))
    d_num = parser.d_num
    im_size = config['FIXED_SIZES'][d_num - 1]


    START_TIME=time.time()
    reporter=Reporter(parser=parser)
    loader=Loader(parser=parser)

    train_gen, valid_gen, test_gen = loader.return_gen()
    train_steps, valid_steps, test_steps = loader.return_step()
    # ---------------------------model----------------------------------

    input_channel_count = parser.input_channel
    output_channel_count = 4
    first_layer_filter_count = parser.filter

    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count, im_size=im_size, parser=parser)
    
    model = network.get_model()
    # model.compile(loss=dice_coef_loss,optimizer='adam', metrics=[dice])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[dice])

    
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
                        default=48, help='Number of model first_filters')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float,
                        default=0.85, help='Training rate')
    parser.add_argument('-es', '--early_stopping', type=int,
                        default=1, help='early_stopping patience')
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
