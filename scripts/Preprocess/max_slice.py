import SimpleITK as sitk
import tensorflow.python
import numpy as np
import nibabel as nib
import scipy.misc
from tqdm import tqdm_notebook, tqdm
import os
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt  # 描画用
import scipy.interpolate as interpolate
import pandas as pd


def rotate(image, angle, center):
    h, w = image.shape[:2]
    affine = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, affine, (w, h))


def load_data(cid, seg_path, vol_path):
    raw_seg = nib.load(seg_path)
    raw_vol = nib.load(vol_path)

    affine = raw_vol.affine
    raw_vol = raw_vol.get_data()
    raw_seg = raw_seg.get_data()
    return raw_vol, raw_seg


def get_slice_idx(raw_seg):
    rnumber = len(raw_seg[0, 0, :])
    imagefragarray = []
    sliceIndex = []

    # 高さ方向の腎臓、腎臓がんの範囲特定
    for x in range(rnumber):

        imagefragarray = raw_seg[:, :, x]

        if np.where(imagefragarray != 0, True, False).any():
            sliceIndex.append(x)

    return sliceIndex

def get_max_datas(seg_voxel,sliceIndex):
    max_area = 0
    max_rect = 0
    max_idx = 0
    for idx in sliceIndex:
        slice_seg = seg_voxel[:, :, idx]
        contours, hierarchy = cv2.findContours(slice_seg, 1, 2)

        max_id = np.argmax(np.array([cv2.contourArea(cnt) for cnt in contours]))
        max_contour = contours[max_id]
        
        area = cv2.contourArea(max_contour)
        rect = cv2.minAreaRect(max_contour)
        center, size, degree = rect
        # 角度から4座標に変換
        box = cv2.boxPoints(rect)
        if area > max_area:
            max_area = area
            max_rect = rect
            max_idx = idx

    return(max_area,max_rect,max_idx)


def get_cut_area(max_rect,PADDING_SIZE=20):
    center, size, degree = max_rect
    box = cv2.boxPoints(max_rect)

    horizon_rect = (center, size, 0)
    horizon_box = cv2.boxPoints(horizon_rect)

    LX = int(horizon_box[0, 0])
    RX = int(horizon_box[3, 0])
    TY = int(horizon_box[0, 1])
    UY = int(horizon_box[1, 1])

    if LX > RX:
        LX, RX = RX, LX

    if TY > UY:
        TY, UY = UY, TY

    PAD_TY = TY - PADDING_SIZE if (TY - PADDING_SIZE) > 0 else 0
    PAD_UY = UY + PADDING_SIZE

    PAD_LX = LX - PADDING_SIZE if(LX - PADDING_SIZE) > 0 else 0
    PAD_RX = RX + PADDING_SIZE

    return(PAD_LX,PAD_RX,PAD_TY,PAD_UY)


def divide_index(sliceIndex):
    max_diff = 1
    diff = 0
    divide_idx = 0
    for x in range(1, len(sliceIndex)):
        diff = sliceIndex[x] - sliceIndex[x - 1]
        if diff >= max_diff:
            max_diff = diff
            slice1 = sliceIndex[:x]
            slice2 = sliceIndex[x:]
            divide_Idx = sliceIndex[x - 1]
    
    return slice1,slice2




def rotate_cut (slice_im,TY,UY,LX,RX,degree,center,IM_SIZE,sitk_flg=True):
    rotate_a = rotate(slice_im, degree, center)
    clip_im = rotate_a[TY:UY, LX:RX]
    clip_im = cv2.resize(clip_im, dsize=(IM_SIZE, IM_SIZE), interpolation=cv2.INTER_LINEAR)
    
    if sitk_flg:
        clip_im = sitk.GetImageFromArray(clip_im)
    return clip_im


def make_paths(IMAGE_PATH,LABEL_PATH,cid,i,idx,prefix):
    os.makedirs(os.path.join(IMAGE_PATH, 'case_00' + str(cid).zfill(3)),exist_ok=True)
    os.makedirs(os.path.join(LABEL_PATH, 'case_00' + str(cid).zfill(3)),exist_ok=True)
    
    label_path = os.path.join(LABEL_PATH, 'case_00' + str(cid).zfill(3), f"{prefix}_{str(i).zfill(3)}_{idx}_label.mha")
    image_path = os.path.join(IMAGE_PATH, 'case_00' + str(cid).zfill(3), f"{prefix}_{str(i).zfill(3)}_{idx}_image.mha")

    return label_path,image_path



if __name__ == "__main__":
    ROOT_PATH = r"C:\Users\higuchi\Desktop\kits19\data\case_00"
    IMAGE_PATH = r"C:\Users\higuchi\Desktop\LAB\201906_\segmentation\data\image"
    LABEL_PATH = r"C:\Users\higuchi\Desktop\LAB\201906_\segmentation\data\label"
    errorcidtxt = r"C:\Users\higuchi\Desktop\LAB\201906_\segmentation\data\error_cid.txt"

    PADDING_SIZE = 20
    IM_SIZE=256 
    log_df = pd.DataFrame(
        columns=[
            'path',
            'padding_size',
            'area_1',
            'size1_0',
            'size1_1',
            'center1_0',
            'center1_1',
            'degree1',
            'maxcid_1',
            'area_2',
            'size2_0',
            'size2_1',
            'center2_0',
            'center2_1',
            'degree2',
            'maxcid_2' ])

    for cid in tqdm(range(210)):
        seg_path = os.path.join(ROOT_PATH + str(cid).zfill(3), "segmentation.nii.gz")
        vol_path = os.path.join(ROOT_PATH + str(cid).zfill(3), "imaging.nii.gz")

        raw_vol, raw_seg = load_data(cid, seg_path, vol_path)
        sliceIndex = get_slice_idx(raw_seg)
        
        slice1, slice2 =divide_index(sliceIndex)
        
            
        try:
            max_area1, max_rect1, max_idx1=get_max_datas(raw_seg, slice1)
            max_area2, max_rect2, max_idx2=get_max_datas(raw_seg, slice2)
        except:
            print('max_rect is zero at :', cid)
            with open(errorcidtxt, mode='w') as f:
                f.write(str(cid))
            continue

            
        if (max_rect1 == 0) or (max_rect2 == 0):
            print('max_rect is zero at :',cid)
            with open(errorcidtxt, mode='w') as f:
                f.write(str(cid))
            continue
        
        center1, size1, degree1 = max_rect1
        center2, size2, degree2 = max_rect2

        PAD_LX1, PAD_RX1, PAD_TY1, PAD_UY1 = get_cut_area(max_rect1)
        PAD_LX2, PAD_RX2, PAD_TY2, PAD_UY2 = get_cut_area(max_rect2)

        datalist = [seg_path, PADDING_SIZE, max_area1, size1[0], size1[1], center1[0], center1[1], degree1, max_idx1,
                    max_area2, size2[0], size2[1], center2[0], center2[1], degree2, max_idx2]
        
        log_row = pd.Series(datalist, index=log_df.columns)
        log_df = log_df.append(log_row, ignore_index=True)

        for i, idx in enumerate(slice1):
            slice_seg = raw_seg[:, :, idx]
            slice_vol = raw_vol[:, :, idx]
            
            seg = rotate_cut(slice_seg, PAD_TY1, PAD_UY1, PAD_LX1, PAD_RX1, degree1, center1, IM_SIZE)
            vol = rotate_cut(slice_vol, PAD_TY1, PAD_UY1, PAD_LX1, PAD_RX1, degree1, center1, IM_SIZE)
            label_path, image_path = make_paths(IMAGE_PATH,LABEL_PATH, cid, i, idx, "front")
            sitk.WriteImage(seg, label_path, True)
            sitk.WriteImage(vol, image_path, True)



        for i, idx in enumerate(reversed(slice2)):
            slice_seg = raw_seg[:, :, idx]
            slice_vol = raw_vol[:, :, idx]
            
            seg = rotate_cut(slice_seg, PAD_TY2, PAD_UY2, PAD_LX2, PAD_RX2, degree2, center2, IM_SIZE)
            vol = rotate_cut(slice_vol, PAD_TY2, PAD_UY2, PAD_LX2, PAD_RX2, degree2, center2, IM_SIZE)
            
            label_path, image_path = make_paths(IMAGE_PATH,LABEL_PATH, cid, i, idx, "back")
            sitk.WriteImage(seg, label_path, True)
            sitk.WriteImage(vol, image_path, True)

        log_df.to_csv('./0701max_slice.csv')




