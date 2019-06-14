import sys
import os
import numpy as np
import argparse
import SimpleITK as sitk
import datetime
import re
DATE_TIME=datetime.datetime.today().strftime("%m%d")

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile", help="The filename(s) of input image(s).", nargs='+')
    parser.add_argument("-o", "--listfile", help="The filename of the list of extracted slice and organ existence list.", default='listfile_256.txt')
    parser.add_argument("-p", "--plane", help="Plane name.", default="axial", choices=["axial", "coronal", "sagittal"])
    parser.add_argument("-s", "--size", help="Output image size. {SIZE}x{SIZE} image will be generated.", type=int, default=256)
    #nargs→受け取るべき引数の数。
    parser.add_argument("--outfilespec", help="The filename format of extracted slice images. (e.g. 'image{no:02d}A{slice:03d}.mha')", default=["image{no:02d}_{slice:03d}.mha"], nargs='*')
    parser.add_argument("--onehot", help="Convert combined multi-label image to one-hot representation. Note that one-hot vector includes background label.", action='store_true')
    parser.add_argument("-m", "--mask", help="The filename of mask image.")
    parser.add_argument("--fittomask", help="Resize masked image to specified size.", action='store_true')
    parser.add_argument("--nomasking", help="Do not perform masking. label image will be used for only slice range.", action='store_true')
    parser.add_argument("--outval", help="Default pixel value.", default=-1024, type=int)
    args = parser.parse_args()
    return args



def createParentPath(filepath):
    #head->それ以外,tail->末尾
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)


def Resampling(image, newsize, plane, roisize, origin = None, is_label = False):
    ivs = image.GetSpacing()

    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
    else:
        minval = None

    if plane == "sagittal":
        osize = (roisize[0], newsize, newsize)
    elif plane == "coronal":
        osize = (newsize, roisize[1], newsize)
    elif plane == "axial":
        osize = (newsize, newsize, roisize[2])

    ovs = [ vs * s / os for vs, s, os in zip(ivs, roisize, osize) ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(osize)
    
    if origin is not None:
        resampler.SetOutputOrigin(origin)
    else:
        resampler.SetOutputOrigin(image.GetOrigin())

    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(ovs)
    
    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled = resampler.Execute(image)

    return resampled


def GetMinimumValue(image):
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(image)
    return minmax.GetMinimum()

def GetMaximumValue(image):
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(image)
    return minmax.GetMaximum()


def ConvertToOnehot(labelimage):
    labelarry = sitk.GetArrayFromImage(labelimage)
    s = labelimage.GetSize()
    nclasses = labelarry.max() + 1

    b = np.zeros((labelarry.size, nclasses), np.uint8)
    b[np.arange(labelarry.size), np.reshape(labelarry, [-1,])] = 1
    onehotarry = np.reshape(b, [s[2], s[1], s[0], nclasses])

    return onehotarry


def ExtractSlice(image, idx, plane):
    if plane == "sagittal":
        slicearry = image[:, :, idx]
    elif plane == "coronal":
        slicearry = image[:, idx, :]
    elif plane == "axial":
        slicearry = image[idx, :, :]

    return slicearry


def main(args):
    if args.fittomask and args.mask is None:
        print("[ERROR] '-m' option must be specified for '--fittomask'.")
        return
    # print(args.imagefile)
    # print(type(args.imagefile))
    maskimage = None
    roisize = None
    origin = None
    slicerange = None
    if args.mask is not None:
        maskimage = sitk.ReadImage(args.mask)
        statfilter=sitk.LabelStatisticsImageFilter()
        statfilter.Execute(maskimage, maskimage)
        bb = statfilter.GetBoundingBox(1)
        mvs = maskimage.GetSpacing()
        
        if args.plane == "sagittal":
            roisize = (maskimage.GetSize()[0], bb[3]-bb[2], bb[5]-bb[4])
            index = (0, bb[2], bb[4])
            slicerange = (bb[0], bb[1]+1)
        elif args.plane == "coronal":
            roisize = (bb[1]-bb[0], maskimage.GetSize()[1], bb[5]-bb[4])
            index = (bb[0], 0, bb[4])
            slicerange = (bb[2], bb[3]+1)
        elif args.plane == "axial":
            roisize = (bb[1]-bb[0], bb[3]-bb[2], maskimage.GetSize()[2])
            index = (bb[0], bb[2], 0)
            slicerange = (bb[4], bb[5]+1)
        origin = maskimage.TransformIndexToPhysicalPoint(index)


    spacing = None
    imagearrys = []
    labelimage = None
    print('load input images.')
    for i in range(len(args.imagefile)):
        print('  loading {}...'.format(args.imagefile[i]), end='', flush=True)
        image = sitk.ReadImage(args.imagefile[i])
        if maskimage is not None and not args.nomasking:
            if image.GetNumberOfComponentsPerPixel() == 1:
                minval = GetMinimumValue(image)
                if minval < args.outval:
                    minval = args.outval
            else:
                minval = 0
            image = sitk.Mask(image, maskimage, minval)

        is_label = image.GetPixelID() == sitk.sitkUInt8

        print("resampling along {0} plane with {1}x{1}...".format(args.plane, args.size), end='', flush=True)

        if args.fittomask:
            resampled = Resampling(image, newsize = args.size, plane = args.plane, roisize = roisize, origin = origin, is_label = is_label)
        else:
            resampled = Resampling(image, newsize = args.size, plane = args.plane, roisize = image.GetSize(), is_label = is_label)

        imagearrys.append(sitk.GetArrayFromImage(resampled))

        if slicerange is None:
            isize = image.GetSize()
            plane2index = { 'sagittal': isize[0], 'coronal': isize[1], 'axial': isize[2] }
            slicerange = (0, plane2index[args.plane])

        if spacing is None:
            ivs = resampled.GetSpacing()
            plane2spacing = { 'sagittal': (ivs[1], ivs[2]), 'coronal': (ivs[0], ivs[2]), 'axial': (ivs[0], ivs[1]) }
            spacing = plane2spacing[args.plane]

        print('done')

    if len(args.outfilespec) == 1:
        outfilespec = args.outfilespec * len(imagearrys)
    else:
        outfilespec = args.outfilespec

    if len(imagearrys) != len(outfilespec):
        print('[ERROR] Mismatch between the numbers of images and outfilespecs.')
        return

    createParentPath(args.listfile)
    f = open(args.listfile, 'a')

    #sidx->各軸のスライスできるrange
    for sidx in range(slicerange[0], slicerange[1]):
        lines = []
        #imagearrays->CTとSEG。pathの数。
        for iidx in range(len(imagearrys)):
            #"image{no:02d}_{slice:03d}.mha" * imagearraysが入力されている。
            #出力先になる。
            # DATA_TYPE= 'CT' if 'imaging.nii.gz' in args.imagefile[iidx] else 'Segmentation'
            
            input_file_path=args.imagefile[0]
            input_file_path, _ = os.path.split(input_file_path)
            input_num = re.sub(r'.+\\','',input_file_path)

            parentsfilename=fr'C:\Users\higuchi\Desktop\LAB\201906_\segmentation\data\input\{args.plane}\{input_num}\{DATE_TIME}_256\\'
            slicefilename = outfilespec[iidx].format(no=iidx, slice=sidx)
            slicefilename =parentsfilename+slicefilename
            
            #その軸の画像だけ取得する。
            slicearry = ExtractSlice(imagearrys[iidx], sidx, args.plane)
            sliceimage = sitk.GetImageFromArray(slicearry, isVector = (slicearry.shape[-1] > 1))
            sliceimage.SetSpacing(spacing)
            createParentPath(slicefilename)
            sitk.WriteImage(sliceimage, slicefilename, True)

            lines.append(slicefilename)

        line = ' '.join(lines)
        print(line)
        f.write(line+'\n')

    f.close()


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
