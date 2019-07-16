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
        self.use_no_c=parser.use_no_c
        self.c=parser.channel
        self.d_num = parser.d_num
        self.batch_size = parser.batch_size
        self.s=parser.split
        self.im_size =256
        self.train=DataSet(0,150,self.parser.split,self.parser.batch_size)
        self.valid=DataSet(150,180,self.parser.split,self.parser.batch_size)
        self.test=DataSet(180,203,self.parser.split,self.parser.batch_size)


    def return_gen(self,return_path=False):

        self.train_gen = self.generator_with_preprocessing(self.train, self.c,only_cancer=True)
        self.valid_gen= self.generator_with_preprocessing(self.valid,self.c,False)
        self.test_gen = self.generator_with_preprocessing(self.test,self.c,False)

        self.output_gen_path=self.generator_path(self.test, self.batch_size)

        if return_path:
            return self.train_gen, self.valid_gen, self.test_gen,self.output_gen_path
        else:    
            return self.train_gen, self.valid_gen, self.test_gen

    def return_step(self):
        return self.train.cancer_step, self.valid.whole_step, self.test.whole_step


    def m2a(self,path):
        im=sitk.ReadImage(path)
        return sitk.GetArrayFromImage(im)

    #チャンネル数に応じて前後スライスを持ってきて、バッチにする。
    #segならwholeもbatchにするべし
    def make_multi_channel_batch(self,whole:'list',batch:'list',onehot=False,channel=3):
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

  

    def generator_with_preprocessing(self,dataset,channel,only_cancer=False):  # , *input_path

        vol_path=dataset.vol_cancer_path if only_cancer else dataset.vol_path
        seg_path=dataset.seg_cancer_path if only_cancer else dataset.seg_path

        while True:
            for i in range(0, len(vol_path), self.batch_size):
                batch_vol_path = vol_path[i:i + self.batch_size]
                batch_seg_path = seg_path[i:i + self.batch_size]
                
                batch_vol = self.make_multi_channel_batch(dataset.vol_path,batch_vol_path,channel=channel)
                batch_seg = self.make_multi_channel_batch(dataset.seg_path,batch_seg_path,onehot=True,channel=1)
                data=(batch_vol,batch_seg)
                
                yield data

    def generator_path(self,data,batch_size):
        while True:
            for i in range(0, len(data.vol_path), batch_size):
                yield data.vol_path[i:i + batch_size]




class DataSet:
    def __init__(self,stcid,encid,split,batch_size,json_path=config):
      
        self.df=pd.read_csv(json_path['slice_csv'],header=0,index_col=0)
        self.data_path = Path(json_path['dataset_path'][DATA_PATH])
        self.vol_path=self.get_list(stcid,encid,split,'image')[0]
        self.seg_path=self.get_list(stcid,encid,split,'label')[0]
        self.vol_cancer_path,self.seg_cancer_path=self.delete_no_cancer(self.vol_path,self.seg_path)
        self.whole_step=math.ceil(len(self.vol_path)/batch_size)
        self.cancer_step=math.ceil(len(self.vol_cancer_path)/batch_size)
   
    @staticmethod
    def strcid(cid):
        return(f'case_00{str(cid).zfill(3)}')

        
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
    
    def delete_no_cancer(self,vol_path,seg_path):
        vol_path=[str(vol) for vol in vol_path]
        seg_path=[str(seg) for seg in seg_path]
        vol_path2=[vol for vol in vol_path if re.search((f'_2_image'),vol)]
        seg_path2=[seg for seg in seg_path if re.search((f'_2_label'),seg)]
        return vol_path2,seg_path2











