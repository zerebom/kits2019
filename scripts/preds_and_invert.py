from Preprocess.affine_slice import load_data,get_slice_idx
from tensorflow.python.keras import losses as kl
from tensorflow.python.keras.models import load_model
import SimpleITK as sitk
import tensorflow.python
import numpy as np
import nibabel as nib
import scipy.misc
from tqdm import tqdm_notebook,tqdm
import os
from dice_coefficient import dice, dice_1, dice_2, dice_coef_loss, DiceLossByClass
import json


import pandas as pd 


def revert_image(square_img, PADDING_SIZE, roi_shape, raw_shape, degree, center):
    rows,cols =roi_shape

    tmp2 = cv2.resize(square_img, dsize=(cols, rows), interpolation=cv2.INTER_CUBIC)
    tmp3 = tmp2[PADDING_SIZE:-PADDING_SIZE + 2, PADDING_SIZE:-PADDING_SIZE + 1]

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -degree, 1)
    M[0, 2] += center[0]
    M[1, 2] += center[1]
    raw_slice = cv2.warpAffine(tmp3, M, (raw_shape[1], raw_shape[0]))

    return raw_slice

def load_vol(vol_path):
    vol_img = sitk.ReadImage(vol_path)
    vol_img = sitk.GetArrayFromImage(vol_img)
    return vol_img
    
if __name__ == "__main__":        
    day='0626'
    df=pd.read_csv('./log_df2.csv')
    config = json.load(open('./setting.json'))
    for d_num in tqdm(range(10)):

        im_size = config['FIXED_SIZES'][d_num]
        loss_function=DiceLossByClass(im_size, 3).dice_coef_loss
        kl.custom_loss = DiceLossByClass(im_size, 3).dice_coef_loss


        model = load_model(f'./Result/Trained_weights/{d_num}/{day}.h5',
        custom_objects={
                        'dice_coef_loss':loss_function,
                        'dice':dice,
                        'dice_1':dice_1,
                        'dice_2':dice_2,
                        })

                # vol_img=load_vol(path,loss=[], optimizer=optimizer, metrics=[dice, dice_1, dice_2])

        d_num_df=df[df['path'].str.contains(rf'\\{d_num}\\')]
        #ほんとは正規表現したい
        # d_num_df=d_num_df[d_num_df['path'].str.contains(rf'\\case_00[]\\')]

        for cid in range(180,210):
            d_num_df=d_num_df[d_num_df['path'].str.contains(rf'\\case_00{str(cid).zfill(3)}\\')]
            for index,row in d_num_df.iterrows():
                path=row.path
                path=path.replace('label_sagittal2','image_sagittal2')

                vol_img=load_vol(path)
                seg_pred=model.predict(vol_img, batch_size=1)
                seg_pred=np.argmax(seg_pred,axis=1)

                roi_shape=(row.roi_s0,row.roi_s1)
                raw_shape=(row.raw_s0,row.raw_s1)
                center=(row.center0,row.center1)
                degree=row.degree
                PADDING_SIZE=row.padding_size

                raw_seg=revert_image(seg_pred, PADDING_SIZE, roi_shape, raw_shape, degree, center)
                save_path=path.replace('image_sagittal','pred_sagittal2')
                dirname,filename=os.path.split(save_path)
                
                filename=re.sub('lable','',filename)
                filename=re.sub('.mha','',filename)

                save_path=os.path.join(dirname,filename.zfill(3)+'.mha')

                raw_seg = sitk.GetImageFromArray(raw_seg)

                sitk.WriteImage(raw_seg, save_path, True)

