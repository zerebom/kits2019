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
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from Models.Unet8 import UNet
from Utils.dice_coefficient import dice, dice_1, dice_2, dice_coef_loss, DiceLossByClass

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
gpu_count = 2

def train(parser):
    config = json.load(open('./setting.json'))
    d_num = parser.d_num
    im_size = config['FIXED_SIZES'][d_num]
    weight_save_dir = config['weight_save_dir']
    weight_file = os.path.join(weight_save_dir, str(d_num), dt.today().strftime("%m%d") + '.h5')



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
        loss=[DiceLossByClass(im_size, 3).dice_coef_loss], optimizer=optimizer, metrics=[dice, dice_1, dice_2])

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
    # mc_cb = ModelCheckpoint(filepath=weight_file, monitor='val_loss', verbose=1, save_best_only=True,
    #                         save_weights_only=False, mode='min', period=1)

    print("start training.")
    print(valid_steps)
    # Pythonジェネレータ（またはSequenceのインスタンス）によりバッチ毎に生成されたデータでモデルを訓練します．
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        max_queue_size=10 if ON_WIN else 32,
        workers=1 if ON_WIN else WORKERS * gpu_count,
        use_multiprocessing=False,
        validation_steps=valid_steps,
        validation_data=valid_gen,
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
