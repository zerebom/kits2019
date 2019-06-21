from PIL import Image
import glob
import os
import json
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import numpy as np
import math
import random
import SimpleITK as sitk
import cv2

config = json.load(open('./setting.json'))
D_num = 1


class Loader:
    def __init__(self, json_path=config, parser=None):
        self.parser = parser
        self.vol_path = json_path['dataset_path']['sagittal_image']
        self.seg_path = json_path['dataset_path']['sagittal_label']
        self.d_num = parser.d_num

        
        self.add_member()
        self.batch_size = parser.batch_size
        self.im_size = json_path['FIXED_SIZES'][self.d_num - 1]

    def add_member(self):
        """
        jsonファイルに記載されている、pathをクラスメンバとして登録する。
        self.Left_RGBとかが追加されている。
        """
        self.train_vol_list, self.valid_vol_list, self.test_vol_list = [], [], []
        self.train_seg_list, self.valid_seg_list, self.test_seg_list = [], [], []

        
        for i in range(150):
            self.train_vol_list += glob.glob(os.path.join(self.vol_path, f'case_00{str(i).zfill(3)}/{self.d_num}/*.mha'))
            self.train_seg_list += glob.glob(os.path.join(self.seg_path, f'case_00{str(i).zfill(3)}/{self.d_num}/*.mha'))

        for i in range(150,180):
            self.valid_vol_list += glob.glob(os.path.join(self.vol_path, f'case_00{str(i).zfill(3)}/{self.d_num}/*.mha'))
            self.valid_seg_list += glob.glob(os.path.join(self.seg_path, f'case_00{str(i).zfill(3)}/{self.d_num}/*.mha'))
        
        for i in range(180,210):
            self.test_vol_list += glob.glob(os.path.join(self.vol_path, f'case_00{str(i).zfill(3)}/{self.d_num}/*.mha'))
            self.test_seg_list += glob.glob(os.path.join(self.seg_path, f'case_00{str(i).zfill(3)}/{self.d_num}/*.mha'))


    def return_gen(self):
        # self.imgs_length = len(self.sagittal_image)
        # self.train_list, self.valid_list, self.test_list = self.train_valid_test_splits(self.imgs_length)

        self.train_steps = math.ceil(len(self.train_seg_list) / self.batch_size)
        self.valid_steps = math.ceil(len(self.valid_seg_list) / self.batch_size)
        self.test_steps = math.ceil(len(self.test_seg_list) / self.batch_size)

        self.train_gen = self.generator_with_preprocessing(self.train_vol_list, self.train_seg_list, self.batch_size)
        self.valid_gen = self.generator_with_preprocessing(self.valid_vol_list, self.train_seg_list, self.batch_size)
        self.test_gen = self.generator_with_preprocessing(self.test_vol_list, self.train_seg_list, self.batch_size)
        return self.train_gen, self.valid_gen, self.test_gen

    def return_step(self):
        return self.train_steps, self.valid_steps, self.test_steps

    @staticmethod
    def train_valid_test_splits(imgs_length: 'int', train_rate=0.8, valid_rate=0.1, test_rate=0.1):
        data_array = list(range(imgs_length))
        tr = math.floor(imgs_length * train_rate)
        vl = math.floor(imgs_length * (train_rate + valid_rate))

        random.shuffle(data_array)
        train_list = data_array[:tr]
        valid_list = data_array[tr:vl]
        test_list = data_array[vl:]

        return train_list, valid_list, test_list

    def load_batch_img_array(self, batch_vol,batch_seg, prepro_callback=False):
        seg_img_list = []
        vol_img_list = []
        for vol,seg in zip(batch_vol,batch_seg):
            vol = sitk.ReadImage(vol)
            seg = sitk.ReadImage(seg)

            vol_img = sitk.GetArrayFromImage(vol)
            seg_img = sitk.GetArrayFromImage(seg)
            
            #データサイズが違ったときはリサイズ
            if vol_img.shape != (self.im_size, self.im_size):
                vol_img = cv2.resize(vol_img, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_CUBIC)

            if seg_img.shape != (self.im_size, self.im_size):
                seg_img = cv2.resize(seg_img, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_CUBIC)

            vol_img_list.append(vol_img)
            seg_img_list.append(seg_img)

        vol_img_list = np.stack(vol_img_list)
        seg_img_list = np.stack(seg_img_list)

        #4次元テンソルにしている
        vol_img_list = vol_img_list[:, : , : , np.newaxis]
        # seg_img_list = seg_img_list[:, :, :, np.newaxis]
        # print(seg_img_list.shape)
        from keras.utils import np_utils
        seg_img_list = np_utils.to_categorical(seg_img_list, num_classes=4)
        # print(seg_img_list.shape)



        return vol_img_list, seg_img_list

    def generator_with_preprocessing(self, vol_path_list,seg_path_list, batch_size):  # , *input_paths
        while True:
            for i in range(0, len(vol_path_list), batch_size):
                batch_vol = vol_path_list[i:i + batch_size]
                batch_seg = seg_path_list[i:i + batch_size]
                batch_vol, batch_seg = self.load_batch_img_array(batch_vol,batch_seg)

                yield(batch_vol, batch_seg)
