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


def transform_img(slice_seg, slice_vol, IM_SIZE):
    contours, hierarchy = cv2.findContours(slice_seg, 1, 2)

    # 最大領域の取得
    max_id = np.argmax(np.array([cv2.contourArea(cnt) for cnt in contours]))
    max_countor = contours[max_id]

    # 面積
    area = cv2.contourArea(max_countor)
    rect = cv2.minAreaRect(max_countor)

    # 回転点,領域の大きさ,回転角
    center, size, degree = rect
    # 角度から4座標に変換
    box = cv2.boxPoints(rect)

    # 角度が0になった場合、関心領域はどこに写像されるか取得
    horizon_rect = (center, size, 0)
    horizon_box = cv2.boxPoints(horizon_rect)

    # (Left,Right,Top,Bottom:X,Y)
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

    # 画像全体を回転させる。
    rotate_vol_a = rotate(slice_vol, degree, center)
    rotate_seg_a = rotate(slice_seg, degree, center)

    clip_vol = rotate_vol_a[PAD_TY:PAD_UY, PAD_LX:PAD_RX]
    clip_seg = rotate_seg_a[PAD_TY:PAD_UY, PAD_LX:PAD_RX]
    box_shape = clip_seg.shape

    clip_vol = cv2.resize(clip_vol, dsize=(IM_SIZE, IM_SIZE), interpolation=cv2.INTER_LINEAR)
    clip_seg = cv2.resize(clip_seg, dsize=(IM_SIZE, IM_SIZE), interpolation=cv2.INTER_LINEAR)

    return clip_seg, clip_vol, box_shape, degree, center, area


def make_paths(cid, folder_num, slice_num):
    label_path = os.path.join(
        r"C:\Users\higuchi\Desktop\LAB\201906_\segmentation\data\label_sagittal2\case_00" + str(cid).zfill(3),
        str(folder_num))

    image_path = os.path.join(
        r"C:\Users\higuchi\Desktop\LAB\201906_\segmentation\data\image_sagittal2\case_00" + str(cid).zfill(3),
        str(folder_num))

    os.makedirs(label_path, exist_ok=True); os.makedirs(image_path, exist_ok=True)

    seg_path = os.path.join(label_path, "label{}.mha".format(slice_num))
    vol_path = os.path.join(image_path, "image{}.mha".format(slice_num))

    seg_jpg_path = os.path.join(label_path, "label{}.jpg".format(slice_num))
    vol_jpg_path = os.path.join(image_path, "image{}.jpg".format(slice_num))

    return seg_path, vol_path, seg_jpg_path, vol_jpg_path


if __name__ == "__main__":
    # 周囲の余白の大きさ
    PADDING_SIZE = 20
    # 各スライスの学習データの画像サイズ
    FIXED_SIZES = [128, 256, 256, 256, 128, 128, 256, 256, 256, 128]
    ROOT_PATH = r"C:\Users\higuchi\Desktop\kits19\data\case_00"

    log_df = pd.DataFrame(
        columns=[
            'path',
            'padding_size',
            'roi_s0',
            'roi_s1',
            'raw_s0',
            'raw_s1',
            'area',
            'degree',
            'center0',
            'center1'])

    for cid in tqdm(range(210)):
        seg_path = os.path.join(ROOT_PATH + str(cid).zfill(3), "segmentation.nii.gz")
        vol_path = os.path.join(ROOT_PATH + str(cid).zfill(3), "imaging.nii.gz")

        raw_vol, raw_seg = load_data(cid, seg_path, vol_path)
        sliceIndex = get_slice_idx(raw_seg)
        each_folder_num = int(len(sliceIndex) // 10)  # 特定されたスライスの枚数/10
        folder_num = 0  # スライスを10等分するための定数

        for i, x in enumerate(sliceIndex):
            if (i % each_folder_num == 0) and (i != 0):
                # 9番目は10で割った余りも含めてフォルダに入れる
                if folder_num != 9:
                    folder_num += 1

            IM_SIZE = FIXED_SIZES[folder_num]

            slice_vol = raw_vol[:, :, x]
            slice_seg = raw_seg[:, :, x]
            slice_seg = slice_seg.astype(np.uint8)

            clip_seg, clip_vol, clip_seg_shape, degree, center, area = transform_img(slice_seg, slice_vol, IM_SIZE)
            seg_path, vol_path, seg_jpg_path, vol_jpg_path = make_paths(cid, folder_num, i)

            if len(clip_seg[clip_seg == 3]) != 0:
                raise Exception

            clip_seg_itk = sitk.GetImageFromArray(clip_seg)
            clip_vol_itk = sitk.GetImageFromArray(clip_vol)
            
            # plt.figure()
            # plt.imshow(clip_seg, cmap='Greys_r', origin='lower')
            # plt.savefig(seg_jpg_path)
            # plt.close()

            # plt.figure()
            # plt.imshow(clip_vol, cmap='Greys_r', origin='lower')
            # plt.savefig(vol_jpg_path)
            # plt.close()

            sitk.WriteImage(clip_seg_itk, seg_path, True)
            sitk.WriteImage(clip_vol_itk, vol_path, True)

            # sitk.WriteImage(clip_seg, seg_jpg_path, True)
            # sitk.WriteImage(clip_vol, vol_jpg_path, True)

            datalist = [seg_path, PADDING_SIZE, clip_seg_shape[0], clip_seg_shape[1],
                        slice_seg.shape[0], slice_seg.shape[1], area,
                        degree, center[0], center[1]]
            log_row = pd.Series(datalist, index=log_df.columns)

            log_df = log_df.append(log_row, ignore_index=True)

    log_df.to_csv('./log_df2.csv')
