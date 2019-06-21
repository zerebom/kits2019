from PIL import Image
import numpy as np
from datetime import datetime as dt
import os
from statistics import mean, median, variance, stdev
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator


class Reporter:
    ROOT_DIR = "Result"
    IMAGE_DIR = "image"
    LEARNING_DIR = "learning"
    INFO_DIR = "info"
    MODEL_DIR = "model"
    PARAMETER = "parameter.txt"
    IMAGE_PREFIX = "epoch_"
    IMAGE_EXTENSION = ".png"

    def __init__(self, result_dir=None, parser=None):
        self._root_dir = self.ROOT_DIR
        self.create_dirs()
        self.parameters = list()
        self.parser = parser
    # def make_main_dir(self):

    def add_model_name(self, model_name):
        if not type(model_name) is str:
            raise ValueError('model_name is not str.')

        self.model_name = model_name

    def add_val_loss(self, val_loss):
        self.val_loss = str(round(min(val_loss)))

    def generate_main_dir(self):
        main_dir = self.val_loss + '_' + dt.today().strftime("%Y%m%d_%H%M") + '_' + self.model_name
        self.main_dir = os.path.join(self._root_dir, main_dir)
        os.makedirs(self.main_dir, exist_ok=True)

    def create_dirs(self):
        os.makedirs(self._root_dir, exist_ok=True)

    def plot_history(self, history, title='loss'):
        # 後でfontsize変える
        plt.rcParams['axes.linewidth'] = 1.0  # axis line width
        plt.rcParams["font.size"] = 24  # 全体のフォントサイズが変更されます。
        plt.rcParams['axes.grid'] = True  # make grid
        plt.plot(history.history['loss'], linewidth=1.5, marker='o')
        plt.plot(history.history['val_loss'], linewidth=1., marker='o')
        plt.tick_params(labelsize=20)

        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='upper right', fontsize=18)
        plt.tight_layout()

        plt.savefig(os.path.join(self.main_dir, title + self.IMAGE_EXTENSION))
        if len(history.history['val_loss']) >= 10:
            plt.xlim(10, len(history.history['val_loss']))
            plt.ylim(0, int(history.history['val_loss'][9] * 1.1))

        plt.savefig(os.path.join(self.main_dir, title + '_remove_outlies_' + self.IMAGE_EXTENSION))

    def add_log_documents(self, add_message):
        self.parameters.append(add_message)

    def save_params(self, history):

        #early_stoppingを考慮
        self.parameters.append("Number of epochs:" + str(len(history.history['val_loss'])))
        self.parameters.append("Batch size:" + str(self.parser.batch_size))
        self.parameters.append("Training rate:" + str(self.parser.trainrate))
        self.parameters.append("Augmentation:" + str(self.parser.augmentation))
        self.parameters.append("min_val_loss:" + str(min(history.history['val_loss'])))
        self.parameters.append("min_loss:" + str(min(history.history['loss'])))

        # self.parameters.append("L2 regularization:" + str(parser.l2reg))
        output = "\n".join(self.parameters)
        filename = os.path.join(self.main_dir, self.PARAMETER)

        with open(filename, mode='w') as f:
            f.write(output)


    def plot_predict(
            self,
            batch_input,
            batch_pred,
            batch_teach: '4dim_array',
            save_folder='train',
            batch_num=1) -> 'im':
        for i in range(batch_input.shape[0]):
            os.makedirs(os.path.join(self.main_dir, save_folder), exist_ok=True)

            seg=np.hstack((batch_pred[i, :, :, :],batch_teach[i, :, :, :]))
            vol =batch_input[i, :, :, :]
            
            plt.figure()
            plt.imshow(seg, cmap='Greys_r', origin='lower')
            plt.savefig(os.path.join(self.main_dir,save_folder,f'pred_{self.parser.batch_size*batch_num+i}.png'))
            
            plt.figure()
            plt.imshow(vol, cmap='Greys_r', origin='lower')
            plt.savefig(os.path.join(self.main_dir, save_folder, f'input_{self.parser.batch_size*batch_num+i}.png'))


