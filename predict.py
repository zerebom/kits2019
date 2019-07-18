import argparse
# import keras.callbacks
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
from Models.UNet8 import UNet8
from Models.UNet5 import UNet5
from Models.UNet4 import UNet4
from Models.UNet7 import UNet7
from Utils.dice_coefficient import dice, dice_1, dice_2, dice_coef_loss, DiceLossByClass
from tensorflow.python.keras.utils import Sequence, multi_gpu_model, plot_model
import time
from tensorflow.python.keras.layers import Input
import tensorflow as tf
from datetime import datetime as dt
from tensorflow import keras
from Utils.reporter import Reporter
from Utils.loader3 import Loader
from Utils.status import ON_WIN
import json
import glob
from keras.layers.core import Lambda
from tensorflow.python.keras.models import load_model


"""
python3 predict.py -c 3 -p 0716_unet8_c3f32.h5
"""
def predict(parser):
    config = json.load(open('./setting.json'))
    d_num = 1
    weight_save_dir = config['weight_save_dir']
    weight_dir = os.path.join(weight_save_dir, str(d_num))
    best_model_path=weight_dir+'/'+parser.path
    reporter = Reporter(parser=parser)
    loader = Loader(parser=parser)


    model=load_model(best_model_path,custom_objects={
        'dice':dice,
        'dice_1':dice_1,
        'dice_2':dice_2,
    })

    print('make output_predict')

    _,_, test_steps = loader.return_step()
    _, _, test_gen, output_gen_path = loader.return_gen(return_path=True)
    reporter.generate_main_dir2(f'./output/{parser.path}')

    for i in range(test_steps):
        print(i,end='')
        batch_output_path = next(output_gen_path)
        batch_input, _ = next(test_gen)
        batch_preds = model.predict(batch_input)
        reporter.output_predict(batch_preds,batch_output_path,suffix=parser.suffix)



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    #関係ない
    parser.add_argument('-s', '--suffix', default='',type=str)
    parser.add_argument('--d_num',type=int,default=1)
    parser.add_argument('-b','--batch_size',type=int,default=32)
    
    parser.add_argument('-sp', '--split',type=int,default=1,help='split')
    parser.add_argument('-c', '--channel',type=int,default=1,help='channel')
    
    parser.add_argument('-un', '--use_no_c',action='store_false',help='use_no_cancer')

    return parser

          
if __name__ == '__main__':
    parser = get_parser().parse_args()
    predict(parser)
