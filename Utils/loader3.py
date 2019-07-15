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
from keras.utils.np_utils import to_categorical
import pandas as pd
import re
from pathlib import Path

config = json.load(open('./setting.json'))
D_num = 1

DATA_PATH='max_sagittal'
class Loader:
    def __init__(self, json_path=config, parser=None):
        self.parser = parser
        self.data_path = Path(json_path['dataset_path'][DATA_PATH])
        self.vol_path= self.data_path
        self.seg_path= self.data_path
        self.use_no_c=parser.use_no_c
        self.c=parser.channel
        self.d_num = parser.d_num
        self.df=pd.read_csv(json_path['slice_csv'],header=0,index_col=0)
        self.batch_size = parser.batch_size
        self.s=parser.split
        self.im_size =256
        self.define_data_list()

    @staticmethod
    def strcid(cid):
        return(f'case_00{str(cid).zfill(3)}')


    def define_data_list(self):
        #[0]->前半部分,[1]->真ん中,[2]->終わり
        self.train_vol_list = self.get_list(0, 150,self.s,'image')[self.d_num-1]
        self.valid_vol_list=self.get_list(150,180,self.s,'image')[self.d_num-1]
        self.test_vol_list=self.get_list(180,203,self.s,'image')[self.d_num-1]
        
        self.train_seg_list=self.get_list(0,150,self.s,'label')[self.d_num-1]
        self.valid_seg_list=self.get_list(150,180,self.s,'label')[self.d_num-1]
        self.test_seg_list=self.get_list(180,203,self.s,'label')[self.d_num-1]

        self.train_vol_list2,self.train_seg_list2 = self.delete_no_cancer(self.train_vol_list,self.train_seg_list)
        self.valid_vol_list2,self.valid_seg_list2 = self.delete_no_cancer(self.valid_vol_list,self.valid_seg_list)
        self.test_vol_list2,self.test_seg_list2 = self.delete_no_cancer(self.test_vol_list,self.test_seg_list)
        
    #path->self.seg_path or vol
    def get_list(self,st,en,split,type)->'list':
        output_list=[[0]*split]

        for cid in self.df['index'][st:en]:
            scid=self.strcid(cid)
            f_list= sorted(self.data_path.glob(f'{scid}/{type}/front*.mha'))
            b_list= sorted(self.data_path.glob(f'{scid}/{type}/back*.mha'))
            f_list=[str(f) for f in f_list]
            b_list=[str(b) for b in b_list]


            lf,lb=len(f_list),len(b_list)
            
            for i in range(split):
                output_list[i]+=f_list[i*lf//split:(i+1)*lf//split]+b_list[i*lb//split:(i+1)*lb//split]
                #最初に入っていた0を消す
                output_list[i].pop(0)
        
        return output_list

            
        


    def return_gen(self,return_path=False):


        self.train_steps = math.ceil(len(self.train_seg_list2) / self.batch_size)
        self.valid_steps = math.ceil(len(self.valid_seg_list2) / self.batch_size)
        self.test_steps = math.ceil(len(self.test_seg_list2) / self.batch_size)

        self.train_gen = self.generator_with_preprocessing(self.train_vol_list, self.train_seg_list,self.train_vol_list2, self.train_seg_list2, self.batch_size,self.c,self.use_no_c)
        self.valid_gen= self.generator_with_preprocessing(self.valid_vol_list, self.valid_seg_list, self.valid_vol_list2, self.valid_seg_list2,self.batch_size,self.c,self.use_no_c)
        self.test_gen = self.generator_with_preprocessing(self.test_vol_list, self.test_seg_list, self.test_vol_list2, self.test_seg_list2,self.batch_size,self.c,self.use_no_c)

        self.output_gen_path=self.generator_paths(self.test_vol_list2, self.batch_size)

        if return_path:
            return self.train_gen, self.valid_gen, self.test_gen,self.output_gen_path
        else:    
            return self.train_gen, self.valid_gen, self.test_gen

    def return_step(self):
        return self.train_steps, self.valid_steps, self.test_steps

    def m2a(self,path):
        im=sitk.ReadImage(path)
        return sitk.GetArrayFromImage(im)

    #チャンネル数に応じて前後スライスを持ってきて、バッチにする。
    #segならwholeもbatchにするべし
    def make_batch(self,whole:'list',batch:'list',onehot=False,channel=3):
        output=[]
        for i in range(len(batch)):
            mini=[0]*channel
            for j,k in zip(range(channel),range(-(channel//2),(channel//2)+1)):
                if 0<= i+k <len(batch):
                    mini[j]=self.m2a(str(whole[whole.index(batch[i+k])]))
                else:
                    mini[j]=self.m2a(str(whole[whole.index(batch[i])]))
            output.append(to_categorical(mini[0],num_classes=3) if onehot else np.stack(mini,axis=-1))
        return(np.stack(output))

    def delete_no_cancer(self,vol_paths,seg_paths):
        vol_paths=[str(vol) for vol in vol_paths]
        seg_paths=[str(seg) for seg in seg_paths]
        vol_paths2=[vol for vol in vol_paths if re.search((f'_2_image'),vol)]
        seg_paths2=[seg for seg in seg_paths if re.search((f'_2_label'),seg)]
        return vol_paths2,seg_paths2

    def generator_with_preprocessing(self, vol_paths, seg_paths,vol_paths2,seg_paths2, batch_size,channel,only_cancer=False):  # , *input_paths
        while True:
            for i in range(0, len(vol_paths2), batch_size):
                batch_vol_paths = vol_paths2[i:i + batch_size]
                batch_seg_paths = seg_paths2[i:i + batch_size]
                
                batch_vol = self.make_batch(vol_paths,batch_vol_paths,channel=channel)
                batch_seg = self.make_batch(seg_paths,batch_seg_paths,onehot=True,channel=1)
                data=(batch_vol,batch_seg)
                
                yield data

    def generator_paths(self,vol_path_list,batch_size):
        while True:
            for i in range(0, len(vol_path_list), batch_size):
                batch_vol_path = vol_path_list[i:i + batch_size]
                batch_output_paths=[re.sub('image','preds',path) for path in batch_vol_path]
                yield batch_output_paths

















