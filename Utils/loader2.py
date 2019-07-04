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
from keras.utils import np_utils
import pandas as pd
import re
config = json.load(open('./setting.json'))
D_num = 1




class Loader:
    def __init__(self, json_path=config, parser=None):
        self.parser = parser
        self.vol_path = json_path['dataset_path']['image']
        self.seg_path = json_path['dataset_path']['label']
        self.d_num = parser.d_num
        self.df=pd.read_csv(json_path['slice_csv'],header=0,index_col=0)
        self.batch_size = parser.batch_size
        self.im_size =256
        
        #[0]->前半部分,[1]->真ん中,[2]->終わり
        self.train_vol_list = self.get_list(0, 150, 'image')[self.d_num-1]
        self.valid_vol_list=self.get_list(150,180,'image')[self.d_num-1]
        self.test_vol_list=self.get_list(180,203,'image')[self.d_num-1]
        
        self.train_seg_list=self.get_list(0,150,'label')[self.d_num-1]
        self.valid_seg_list=self.get_list(150,180,'label')[self.d_num-1]
        self.test_seg_list=self.get_list(180,203,'label')[self.d_num-1]


        
        
    def get_list(self,st,en,dtype):
        '''
        dtype:imageかlabel,
        st,en,患者idの始まりと終わり
        '''
        # sep_df=self.df['index'][:150] if d_num==0 else self.df['index'][150:180] if d_num==1 self.df['index'][180:]
        list1, list2, list3=[],[],[]
        for i,cid in enumerate(self.df['index'][st:en]):
            if parser.select_area_size:
                if (self.df.at[i, 'fixed_area1'] < 10000) and (self.df.at[i, 'fixed_area1'] > 4000):
                    f_list=[p for p in glob.glob(f'./data/{dtype}/case_00{str(cid).zfill(3)}/*') if re.search((f'front'),p)]
                if (self.df.at[i,'fixed_area2'] < 10000 and self.df.at[i,'fixed_area2'] > 4000):
                    b_list=[p for p in glob.glob(f'./data/{dtype}/case_00{str(cid).zfill(3)}/*') if re.search((f'back'),p)]  

            else:                
                b_list=[p for p in glob.glob(f'./data/{dtype}/case_00{str(cid).zfill(3)}/*') if re.search((f'back'),p)]  
                f_list=[p for p in glob.glob(f'./data/{dtype}/case_00{str(cid).zfill(3)}/*') if re.search((f'front'),p)]

            b_list=b_list.sort()    
            f_list=f_list.sort()    


            
            lf,lb=len(f_list),len(b_list)
            
            f_list1,f_list2,f_list3=f_list[:lf//3],f_list[lf//3:2*lf//3],f_list[2*lf//3:]
            b_list1,b_list2,b_list3=b_list[:lb//3],b_list[lb//3:2*lb//3],b_list[2*lb//3:]
            
            list1+=f_list1+b_list1
            list2+=f_list2+b_list2
            list3+=f_list3+b_list3

        return list1,list2,list3
        


    def return_gen(self):
        # self.imgs_length = len(self.sagittal_image)
        # self.train_list, self.valid_list, self.test_list = self.train_valid_test_splits(self.imgs_length)

        self.train_steps = math.ceil(len(self.train_seg_list) / self.batch_size)
        self.valid_steps = math.ceil(len(self.valid_seg_list) / self.batch_size)
        self.test_steps = math.ceil(len(self.test_seg_list) / self.batch_size)

        self.train_gen = self.generator_with_preprocessing(self.train_vol_list, self.train_seg_list, self.batch_size)
        self.valid_gen = self.generator_with_preprocessing(self.valid_vol_list, self.valid_seg_list, self.batch_size)
        self.test_gen = self.generator_with_preprocessing(self.test_vol_list, self.test_seg_list, self.batch_size)
        return self.train_gen, self.valid_gen, self.test_gen

    def return_step(self):
        return self.train_steps, self.valid_steps, self.test_steps

    

    def load_batch_img_array(self, batch_vol_path, batch_seg_path, prepro_callback=False):
        seg_img_list = []
        vol_img_list = []
        for vol, seg in zip(batch_vol_path, batch_seg_path):
            vol_img = sitk.ReadImage(vol)
            seg_img = sitk.ReadImage(seg)

            vol_img = sitk.GetArrayFromImage(vol_img)
            seg_img = sitk.GetArrayFromImage(seg_img)
   

            # データサイズが違ったときはリサイズ
            if vol_img.shape != (self.im_size, self.im_size):
                vol_img = cv2.resize(vol_img, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_CUBIC)

            if seg_img.shape != (self.im_size, self.im_size):
                seg_img = cv2.resize(seg_img, dsize=(self.im_size, self.im_size), interpolation=cv2.INTER_CUBIC)


            seg_img = np.clip(seg_img, 0, 2)
            seg_img = np_utils.to_categorical(seg_img, num_classes=3)

            vol_img_list.append(vol_img)
            seg_img_list.append(seg_img)

        vol_img_list = np.stack(vol_img_list)
        seg_img_list = np.stack(seg_img_list)

        # 4次元テンソルにしている
        vol_img_list = vol_img_list[:, :, :, np.newaxis]
        return vol_img_list, seg_img_list

    def generator_with_preprocessing(self, vol_path_list, seg_path_list, batch_size):  # , *input_paths
        while True:
            for i in range(0, len(vol_path_list), batch_size):
                batch_vol_path = vol_path_list[i:i + batch_size]
                batch_seg_path = seg_path_list[i:i + batch_size]
                batch_vol_path, batch_seg_path = self.load_batch_img_array(batch_vol_path, batch_seg_path)

                yield(batch_vol_path, batch_seg_path)
















