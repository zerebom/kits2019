from PIL import Image
import numpy as np
from datetime import datetime as dt
import os
from statistics import mean, median, variance, stdev
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import re
import SimpleITK as sitk
from datetime import datetime as dt
from pathlib import Path

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
        self.d_num = parser.d_num
    # def make_main_dir(self):

    def add_model_name(self, model_name):
        if not type(model_name) is str:
            raise ValueError('model_name is not str.')

        self.model_name = model_name

    def add_val_loss(self, val_loss):
        self.val_loss = str(round(min(val_loss),2))

    def add_val_dice(self, val_dice):
        self.val_dice = str(round(max(val_dice),2))

    def generate_main_dir(self):
        # main_dir = self.val_loss + '_' + dt.today().strftime("%Y%m%d_%H%M") + '_' + self.model_name
        main_dir = str(self.d_num) + '_' + self.val_dice + '_' + dt.today().strftime("%Y%m%d_%H%M") + '_' + self.parser.suffix
        self.main_dir = Path(os.path.join(self._root_dir, main_dir))
        os.makedirs(str(self.main_dir), exist_ok=True)

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

        plt.figure()
        plt.plot(history.history['dice'], linewidth=1.5, marker='o')
        plt.plot(history.history['val_dice'], linewidth=1., marker='o')
        plt.tick_params(labelsize=20)
        plt.title('model dice')
        plt.xlabel('epoch')
        plt.ylabel('dice')
        plt.legend(['dice', 'val_dice'], loc='lower right', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(self.main_dir, 'dice' + self.IMAGE_EXTENSION))



    def add_log_documents(self, add_message):
        self.parameters.append(add_message)

    def save_params(self, history):

        #early_stoppingを考慮
        self.parameters.append("Number of epochs:" + str(len(history.history['val_loss'])))
        self.parameters.append("Batch size:" + str(self.parser.batch_size))
        self.parameters.append("fillter_size:"+str(self.parser.filter))
        self.parameters.append("early stopping patience:"+str(self.parser.early_stopping))
        self.parameters.append("split:"+str(self.parser.split))
        self.parameters.append("input channel:"+str(self.parser.channel))
        self.parameters.append("not use no canser slice:"+str(self.parser.use_no_c))
        self.parameters.append("not use big canser slice:"+str(self.parser.select_area_size))
        self.parameters.append("Training rate:" + str(self.parser.trainrate))
        self.parameters.append("Augmentation:" + str(self.parser.augmentation))
        self.parameters.append("min_val_loss:" + str(min(history.history['val_loss'])))
        self.parameters.append("min_loss:" + str(min(history.history['loss'])))
        if self.parser.select_area_size:self.parameter.append("select_area_size:True")

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
            os.makedirs(str(self.main_dir/save_folder), exist_ok=True)
            
            #one_hot->grayscaleに変換している。
            pred = np.argmax(batch_pred[i, :, :, :],axis=2)
            teach = np.argmax(batch_teach[i, :, :, :],axis=2)

            seg=np.hstack((pred,teach))
            seg = np.where(seg == 1, 128, seg)
            seg = np.where(seg == 2, 255, seg)
            #3次元に展開
            # seg=seg[:,:,np.newaxis]
            seg_im = seg.astype(np.uint8)
            seg_im=Image.fromarray(seg_im,mode='L')

            
            vol =batch_input[i, :, :, :]
            vol=255*vol/(np.max(vol)-np.min(vol))
            vol_im=vol.astype(np.uint8)
            try:
                array_to_img(vol_im).save(str(self.main_dir/save_folder/f'input_{self.parser.batch_size*batch_num+i}.png'))
            except:
                pass

            seg_im.save(str(self.main_dir/save_folder/f'pred_{self.parser.batch_size*batch_num+i}.png'))

    def output_predict(self,batch_preds,batch_input_paths,suffix):
        for pred,path in zip(batch_preds,batch_input_paths):
           
            path=Path(path)
            slice_name=path.name
            cid=path.parents[1].name

            folder=self.main_dir/'preds'/cid
            os.makedirs(str(folder),exist_ok=True)
            
            pred_im=sitk.GetImageFromArray(pred)
            sitk.WriteImage(pred_im,str(folder/slice_name),True)




