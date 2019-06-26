import nibabel as nib
import SimpleITK as sitk
import tensorflow.python
import numpy as np
import scipy.misc

import os

def load_case(cid):
    # Resolve location where data should be living
    data_path = Path(__file__).parent.parent / "data"
    if not data_path.exists():
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )

    # Get case_id from provided cid
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    # Make sure that case_id exists under the data_path
    case_path = data_path / case_id
    if not case_path.exists():
        raise ValueError(
            "Case could not be found \"{}\"".format(case_path.name)
        )

    vol = nib.load(str(case_path / "imaging.nii.gz"))
    seg = nib.load(str(case_path / "segmentation.nii.gz"))
    return vol, seg




#関心領域(seg_dataが存在する部分)の抽出
def extract_ROI(raw_vol,raw_seg,padding=None):
    segbin=np.greater(raw_seg,0)
    repeated_segbin=np.stack((segbin,segbin,segbin),axis=-1)

    seg=np.where(repeated_segbin,raw_seg,0)
    vol=np.where(repeated_segbin,raw_vol,0)

    return seg,vol



def coronal_resize_voxel(vol,seg,affine,raw_size,FIXED_SIZE=None):
    spc_ratio = np.abs(np.sum(affine[2,:]))/np.abs(np.sum(affine[0,:]))
    X,Y,Z=raw_size
    for i in range(Y):
        if np.where(seg[:i:]!=0,True,False).any():
            if FIXED_SIZE:
                vol_im = scipy.misc.imresize(vol[:,i,:], (FIXED_SIZE), interp="bicubic")
                seg_im = scipy.misc.imresize(seg[:,i,:], (FIXED_SIZE),interp="nearest")
        #拡大縮小する対象/#出力のサイズを指定
            else:
                vol_im = scipy.misc.imresize(vol[:,i,:], (int(X*spc_ratio),int(Z)), interp="bicubic")
                seg_im = scipy.misc.imresize(seg[:,i,:], (int(X*spc_ratio),int(Z)),interp="nearest")
            yield vol_im,seg_im

def sagittal_resize_voxel(vol,seg,affine,raw_size,FIXED_SIZE=None):
    spc_ratio = np.abs(np.sum(affine[2,:]))/np.abs(np.sum(affine[0,:]))
    X,Y,Z=raw_size
    
    for i in range(Z):
       #拡大縮小する対象/#出力のサイズを指定
        if FIXED_SIZE:
            vol_im = scipy.misc.imresize(vol[:,i,:], FIXED_SIZE, interp="bicubic")
            seg_im = scipy.misc.imresize(seg[:,i,:], FIXED_SIZE,interp="nearest")
        else:
            vol_im = scipy.misc.imresize(vol[:,:,i], (int(X*spc_ratio),int(Y)), interp="bicubic")
            seg_im = scipy.misc.imresize(seg[:,:,i], (int(X*spc_ratio),int(Y)),interp="nearest")
        
        yield vol_im,seg_im



    

if __name__ == "__main__":
    Global_id=0
    for i in range(210):
        local_id=0

        vol,seg=load_case(i)

        affine = vol.affine
        vol = vol.get_data()
        seg = seg.get_data()
        raw_size=vol.shape

        ROI_vol,ROI_seg=extract_ROI(vol,seg)
        
        for vol,seg in sagittal_resize_voxel(vol,seg,affine,raw_size,(256,256)):
            pass






for q in tqdm_notebook(range(210)):
    ## Read image
    seg = nib.load(os.path.join(r"C:\Users\higuchi\Desktop\kits19\data\case_00"+str(q).zfill(3),"segmentation.nii.gz"))
    vol = nib.load(os.path.join(r"C:\Users\higuchi\Desktop\kits19\data\case_00"+str(q).zfill(3),"imaging.nii.gz"))

    affine = vol.affine
    vol = vol.get_data()
    seg = seg.get_data()

    imagefragarray = []
    imageIndex = []
    count = 0

    #sagittal
    rnumber = len(seg[0,:,0])
    
    #高さ方向の腎臓、腎臓がんの範囲特定
    for x in range(rnumber):
        imagefragarray = seg[:,:,z]
        if np.where(imagefragarray!=0,True,False).any():
            imageIndex.append(z)
    
   # print(len(imageIndex))
    number = int(len(imageIndex)/10)#特定されたスライスの枚数/10
    snumber = -1 #スライスを10等分するための定数
  #  print(number)

    #スライスの保存（imageとsegmentation）
    for i,z in enumerate(imageIndex):
        vol_im = scipy.misc.imresize(vol[:,:,z], FIXED_SIZE, interp="bicubic")
        seg_im = scipy.misc.imresize(seg[:,:,z], FIXED_SIZE,interp="nearest")

        segfrag = sitk.GetImageFromArray(seg_im)
        volfrag = sitk.GetImageFromArray(vol_im)

        if i%number==0 and snumber < 10:
            snumber += 1

       # print("saving cut to", str(q)+"imagefragment"+str(i), end="...", flush=True)

        outfile1 = os.path.join(r"C:\Users\higuchi\Desktop\kits19\data\label\case_00"+str(q).zfill(3),str(snumber),"label{}.mha".format(i))
        outfile2 = os.path.join(r"C:\Users\higuchi\Desktop\kits19\data\image\case_00"+str(q).zfill(3),str(snumber),"image{}.mha".format(i))

        sitk.WriteImage(segfrag, outfile1, True)
        sitk.WriteImage(volfrag, outfile2, True)
        count += 1

