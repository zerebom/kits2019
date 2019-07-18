import keras.backend as KB
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
import os
import glob
import SimpleITK as sitk
import argparse
from tqdm import tqdm
import re
import pandas as pd
from pathlib import Path
from statistics import mean
IMSIZE=(256,256)

def get_parser():
	parser=argparse.ArgumentParser(
	  prog='caluclate dice score for output_slice',
	  usage='python output_dice.py',
	  description='This module caluclate dice score.',
	  add_help=False
	)

	parser.add_argument('-pf','--pred_folder',type=str)
	parser.add_argument('-cs','--csv_suffix',type=str)
	
	return parser

def DICE(turelabel, result):
	intersection=np.sum(np.minimum(np.equal(turelabel,result),turelabel))
	union = np.count_nonzero(turelabel)+np.count_nonzero(result)
	dice = 2 * intersection / union
   # print("intersection: ",2* intersection)
	#print("union: ", union)
	return dice

def averagenum(num):
	if len(num) == 0:
		return 1.0
	
	else: 
		nsum = 0
		for i in range(len(num)):
			nsum += num[i]

		return nsum / len(num)


def calculate_dice(parser):
	df=pd.DataFrame(columns=['cid','whole','kid','cancer'])
	mw_dice,mk_dice,mc_dice=[],[],[]
	for cid in range(182,203):
	# pred_voxel=np.zeros(IMSIZE)
		print(parser.pred_folder)
	
		preds_paths=list(Path(f'./output/{parser.pred_folder}/preds/case_00{str(cid)}').glob('*'))
		preds_paths.sort()
		true_paths=list(Path(f'./data/input/max/sagittal/case_00{str(cid)}/label').glob('*'))
		true_paths.sort()
		print(len(preds_paths),len(true_paths))

		for i,(pred,true) in enumerate(zip(preds_paths,true_paths)):
			pred_img=sitk.GetArrayFromImage(sitk.ReadImage(str(pred)))
			true_img=sitk.GetArrayFromImage(sitk.ReadImage(str(true)))



			pred_voxel=pred_img if i==0 else np.dstack((pred_voxel,pred_img))
			true_voxel=true_img if i==0 else np.dstack((true_voxel,true_img))
			if i==0:continue


		kid_pred=np.where(pred_voxel==1,1,0)
		kid_true=np.where(true_voxel==1,1,0)

		can_pred=np.where(pred_voxel==2,2,0)
		can_true=np.where(true_voxel==2,2,0)


		w_dice=DICE(pred_voxel,true_voxel)
		k_dice=DICE(kid_pred,kid_true)
		c_dice=DICE(can_pred,can_true)

		mw_dice.append(w_dice)
		mk_dice.append(k_dice)
		mc_dice.append(c_dice)

		data_list=[str(cid),w_dice,k_dice,c_dice]
		row = pd.Series(data_list, index=df.columns)
		df = df.append(row, ignore_index=True)

		print(f"""
		Dice of case_00{str(cid).zfill(3)} is...
		whole:{w_dice}
		kidney:{k_dice}
		cancer:{c_dice}

		""")
	
	print(f"""
		mean is...
		whole:{mean(mw_dice)}
		kidney:{mean(mk_dice)}
		cancer:{mean(mc_dice)}

		""")
	mean_list=['mean',mean(mw_dice),mean(mk_dice),mean(mc_dice)]
	mean_row = pd.Series(mean_list, index=df.columns)
	df = df.append(mean_row, ignore_index=True)

	df.to_csv(f'./output/{parser.pred_folder}/{round(mean(mc_dice),3)}_dice.csv')

		





	  



if __name__ == "__main__":
  parser=get_parser().parse_args()
  calculate_dice(parser)