import glob
from scipy.ndimage.interpolation import zoom
import re 
import SimpleITK as sitk
import os
import argparse
import numpy as np
from pathlib import Path

"""
スライスのスペーシングを同じにする
python3 scripts/Preprocess/resample_spacing.py -i ~/Desktop/kits19/data -o ./data/input/image 
"""


def numericalSort(value):
    numbers=re.compile(r'(\d+)')
    parts=numbers.split(value)
    parts[1::2]=map(int,parts[1::2])
    return parts

def saveNPY(array,save_path):
    array=array.astype(np.int16)
    save_path+=".npy"
    np.save(save_path,array)

def resample_spacing(image,new_spacing=[1,1,1],interpolator=sitk.sitkLinear):
    resample=sitk.ResampleImageFilter()
    #補完方法
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(new_spacing)

    new_size=np.array(image.GetSize(),dtype=np.float32)*(np.array(image.GetSpacing(),dtype=np.float32)/new_spacing)
    new_size=np.ceil(new_size).astype(np.int)
    new_size=[int(s)for s in new_size]
    resample.SetSize(new_size)

    return resample.Execute(image)


def main(parser):


    input_dir=Path(parser.input_dir).resolve()
    output_dir=Path(parser.output_dir).resolve()

    for i,imagepath in enumerate(sorted(input_dir.glob("*/imaging.nii.gz"))):

            
        print(imagepath)
        outputfile=output_dir / imagepath.parents[0].stem /'image'
        os.makedirs(outputfile.parents[0],exist_ok=True)
        
        image=sitk.ReadImage(str(imagepath))
        image_re=image
        image_re=resample_spacing(image,[1.0,1.0,1.0]) 
        if parser.save_mha:
            sitk.WriteImage(image_re,str(outputfile)+'.mha',True)
        else:
            image_arr=sitk.GetArrayFromImage(image_re)
            saveNPY(image_arr,str(outputfile))
    
    for labelpath in sorted(input_dir.glob("*/segmentation.nii.gz")):
        outputfile=output_dir / labelpath.parents[0].stem /'label'
        os.makedirs(outputfile.parents[0],exist_ok=True)
        
        label=sitk.ReadImage(str(labelpath))
        label_re=label
        label_re=resample_spacing(label,[1.0,1.0,1.0],interpolator=sitk.sitkNearestNeighbor) 
        if parser.save_mha:
            sitk.WriteImage(label_re,str(outputfile)+'.mha',True)
        else:
            label_arr=sitk.GetArrayFromImage(label_re)
            saveNPY(label_arr,str(outputfile))






def get_parser():
    parser = argparse.ArgumentParser(
        prog='generate parallax image using U-Net',
        usage='python main.py',
        description='This module　generate parallax image using U-Net.',
        add_help=True
    )

    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-m','--save_mha',action='store_true')

                     
    return parser


if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)
