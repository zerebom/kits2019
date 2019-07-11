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
from pathlib import Path

config = json.load(open('./setting.json'))
D_num = 1

DATA_PATH='max_sagittal'
class Loader:
    def __init__(self, json_path=config, parser=None):
        self.parser = parser
        self.data_path = Path(json_path[DATA_PATH])
        self.vol_path= self.data_path/'image'
        self.seg_path= self.data_path/'label'
        self.use_no_c=parser.use_no_c
        self.c=parser.channel

        self.d_num = parser.d_num
        self.df=pd.read_csv(json_path['slice_csv'],header=0,index_col=0)
        self.batch_size = parser.batch_size
        self.s=parser.split
        self.im_size =256

    @staticmethod
    def strcid(cid):
        return(f'case00_{str(cid).zfill(3)}')


    def define_data_list(self):
        #[0]->前半部分,[1]->真ん中,[2]->終わり
        self.train_vol_list = self.get_list(0, 150,self.s,self.vol_path)[self.d_num-1]
        self.valid_vol_list=self.get_list(150,180,self.s,self.vol_path)[self.d_num-1]
        self.test_vol_list=self.get_list(180,203,self.s,self.vol_path)[self.d_num-1]
        
        self.train_seg_list=self.get_list(0,150,self.s,self.seg_path)[self.d_num-1]
        self.valid_seg_list=self.get_list(150,180,self.s,self.seg_path)[self.d_num-1]
        self.test_seg_list=self.get_list(180,203,self.s,self.seg_path)[self.d_num-1]

    #path->self.seg_path or vol
    def get_list(self,st,en,split,path:'Path')->'list':
        output_list=[0]*split

        for cid in self.df['index'][st:en]:
            f_list= sorted(path.glob(self.strcid(cid)/'front*.mha'))
            b_list= sorted(path.glob(self.strcid(cid)/'back*.mha'))
            lf,lb=len(f_list),len(b_list)
            
            for i in range(split):
                output_list[i]+=f_list[i*lf//split:(i+1)*lf//split]+b_list[i*lb//split:(i+1)*lb//split]
        
        return output_list

            
        


    def return_gen(self,return_path=False):
        self.train_steps = math.ceil(len(self.train_seg_list) / self.batch_size)
        self.valid_steps = math.ceil(len(self.valid_seg_list) / self.batch_size)
        self.test_steps = math.ceil(len(self.test_seg_list) / self.batch_size)

        self.train_gen = self.generator_with_preprocessing(self.train_vol_list, self.train_seg_list, self.batch_size,self.c,self.use_no_c)
        self.valid_gen= self.generator_with_preprocessing(self.valid_vol_list, self.valid_seg_list, self.batch_size,self.c,self.use_no_c)
        self.test_gen = self.generator_with_preprocessing(self.test_vol_list, self.test_seg_list, self.batch_size,self.c,self.use_no_c)

        self.output_gen_path=self.generator_paths(self.test_vol_list, self.batch_size)

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
    def make_batch(self,whole:'list',batch:'list',channel=3):
        output=[]
        for i in range(len(batch)):
            mini=[0]*channel
            for j,k in zip(range(channel),range(-(channel//2),(channel//2)+1)):
                if 0<= i+k <len(batch):
                    mini[j]=self.m2a(str(whole[whole.index(batch[i+k])]))
                else:
                    mini[j]=self.m2a(str(whole[whole.index(batch[i])]))
            output.append(np.stack(mini,axis=-1))
        return(np.stack(output))



    def generator_with_preprocessing(self, vol_paths, seg_paths, batch_size,channel,only_cancer=False):  # , *input_paths
        if only_cancer:
            vol_paths2=[vol for vol in vol_paths if re.search((f'_2_label'),vol)]
            seg_paths2=[seg for seg in seg_paths if re.search((f'_2_label'),seg)]
        else:
            vol_paths2=vol_paths
            seg_paths2=seg_paths


        while True:
            for i in range(0, len(vol_paths), batch_size):
                batch_vol_paths = vol_paths2[i:i + batch_size]
                batch_seg_paths = seg_paths2[i:i + batch_size]
                
                batch_vol = self.make_batch(vol_paths,batch_vol_paths,channel=channel)
                batch_seg = self.make_batch(seg_paths,batch_seg_paths,channel=1)
                data=(batch_vol,batch_seg)
                
                yield data

    def generator_paths(self,vol_path_list,batch_size):
        while True:
            for i in range(0, len(vol_path_list), batch_size):
                batch_vol_path = vol_path_list[i:i + batch_size]
                batch_output_paths=[re.sub('image','preds',path) for path in batch_vol_path]
                yield batch_output_paths

















