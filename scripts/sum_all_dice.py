from preprocess.affine_slice import load_data,get_slice_idx
from keras.models import load_model
import SimpleITK as sitk
import tensorflow.python
import numpy as np
import nibabel as nib
import scipy.misc
from tqdm import tqdm_notebook,tqdm
import os
import numpy as np

"""
python3 output_dice.py -pf 1_0.86_20190716_1416_unet7_c1
python3 predict.py 
"""
def load_seg(vol_path):
    vol_img = sitk.ReadImage(vol_path)
    vol_img = sitk.GetArrayFromImage(vol_img)
    return vol_img


def dice_1(preds,teach):
    preds=np.where(preds!=1,0,preds)
    teach=np.where(teach!=1,0,teach)

    eps=0.00001
    union=len(preds[preds!=0])+len(union[union!=0])
    intersection=np.sum(preds==union)

    dice=2*intersection/(union+eps)
    return dice



def dice_2(preds,teach):
    preds=np.where(preds!=1,0,preds)
    teach=np.where(teach!=1,0,teach)

    eps=0.00001
    union=len(preds[preds!=0])+len(union[union!=0])
    intersection=np.sum(preds==union)

    dice=2*intersection/(union+eps)

    return dice


if __name__ == "__main__":
    #linux側にデータがないので手元でやった方が良い。
    ROOT_PATH = r"C:\Users\higuchi\Desktop\kits19\data\case_00"
    PREDS_PATH= "../data/pred_sagittal2"
    dice_1_list=np.zeros(30)
    dice_2_list=np.zeros(30)

    for i,cid in tqdm(enumerate(range(180,210))):
        seg_path = os.path.join(ROOT_PATH + str(cid).zfill(3), "segmentation.nii.gz")
        raw_seg = nib.load(seg_path)
        raw_seg = raw_seg.get_data()

        sliceIndex=get_slice_idx(raw_seg)
        predPaths=glob.glob(os.path.join(PREDS_PATH,str(cid).zfill(3)))
        preds_boxel=np.zeros(shape=raw_seg)

        for idx,pred in zip(sliceIndex,predPaths):
            pred_img=load_vol(pred)
            preds_boxel[:,:,idx]=pred_img

        dice_1_list[i]=dice_1(preds_boxel,raw_seg)
        dice_2_list[i]=dice_2(preds_boxel,raw_seg)

    print(f"""
    dice_1_mean:{np.mean(dice_1_list)}
    dice_1_std:{np.std(dice_1_list)}
    dice_2_mean:{np.mean(dice_2_list)}
    dice_2_std:{np.std(dice_2_list)}
    """)



            





        



    

