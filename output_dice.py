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
import pandas as import pd


IMSIZE=(256,256)

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


def caluclate_dice(parser):
	df=pd.DataFrame(columns=['cid','whole','kid','cancer'])
	mw_dice,mk_dice,mc_dice=[],[],[]
	for cid in tqdm(range(180,203)):
	# pred_voxel=np.zeros(IMSIZE)
		preds_paths=[p for p in glob.glob(parser.pred_folder) if re.search(f'case_00{str(cid)}',p)]
		preds_paths.sort()
		true_paths=[re.sub('preds/*/case_','label/case_') for p in preds_paths]

		for i,(pred,true) in enumerate(zip(preds_paths,true_paths)):
			pred_img=sitk.GetArratFromImage(sitk.ReadImage(pred))
			ture_img=sitk.GetArratFromImage(sitk.ReadImage(ture))

			pred_voxel=np.dstack(pred_voxel,pred_img)
			true_voxel=np.dstack(true_voxel,true_img)


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
		row = pd.Series(data_list, index=log_df.columns)
		df = df.append(row, ignore_index=True)

		print(f"""
		Dice of case_00{str(cid).zfill(3)} is...
		whole:{w_dice}
		kidney:{k_dice}
		cancer:{c_dice}

		""")
	
	print(f"""
		mean is...
		whole:{mw_dice/len(mw_dice)}
		kidney:{mk_dice/len(mk_dice)}
		cancer:{mc_dice/len(mc_dice)}

		""")
	mean=['mean',mw_dice/len(mw_dice),mk_dice/len(mk_dice),mc_dice/len(mc_dice)]
	df.to_csv(f'./output_dice_{parser.csv_suffix}.csv')

		





      








def get_parser():
    parser=argparse.ArgumentParser(
      prog='caluclate dice score for output_slice',
      usage='python output_dice.py',
      description='This module caluclate dice score.',
      add_help=False
    )

    parser.add_argument('-gf','--gtrues_folder',type=str)
    parser.add_argument('-pf','--pred_folder',type=str)
	parser.add_argument('-cs','--csv_suffix',type=str)

if __name__ == "__main__":
  parser=get_parser().parse_args()
  caluclate_dice(parser)

