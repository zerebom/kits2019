import sys
import os
import tensorflow as tf
import numpy as np
import argparse
import SimpleITK as sitk

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile", help="Input image file")
    parser.add_argument("modelfile", help="U-net model file (*.yml).")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("outfile", help="Segmented label file.")
    parser.add_argument("--paoutfile", help="The filename of the estimated probabilistic map file.")
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size", default=1, type=int)
    args = parser.parse_args()
    return args


def createParentPath(filepath):
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.log_device_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    print('loading input image {}...'.format(args.imagefile), end='', flush=True)
    orgimage = sitk.ReadImage(args.imagefile)
    isize = orgimage.GetSize()
    print('done')

    if isize[0] != 256 or isize[1] != 256:
        osize = (256, 256, isize[2])
        ivs = orgimage.GetSpacing()
        ovs = [ isize[i] / osize[i] * ivs[i] for i in range(3)]
        resampled = sitk.Resample(orgimage, osize, sitk.Transform(), sitk.sitkLinear, orgimage.GetOrigin(), ovs, orgimage.GetDirection())
        inputimage = resampled
        imagearry = sitk.GetArrayFromImage(resampled)
    else:
        inputimage = orgimage
        imagearry = sitk.GetArrayFromImage(orgimage)

    imagearry = imagearry[..., np.newaxis]
    print('Shape of input image: {}'.format(imagearry.shape))

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        print('loading U-net model {}...'.format(args.modelfile), end='', flush=True)
        with open(args.modelfile) as f:
            model = tf.keras.models.model_from_yaml(f.read())
        model.load_weights(args.modelweightfile)
        print('done')

    createParentPath(args.outfile)

    print('segmenting...')
    paarry = model.predict(imagearry, batch_size = args.batchsize, verbose = 1)
    #print('paarry.shape: {}'.format(paarry.shape))
    labelarry = np.argmax(paarry, axis=-1).astype(np.uint8)
    #print('labelarry.shape: {}'.format(labelarry.shape))

    print('saving segmented label to {}...'.format(args.outfile), end='', flush=True)
    segmentation = sitk.GetImageFromArray(labelarry)
    segmentation.SetOrigin(inputimage.GetOrigin())
    segmentation.SetSpacing(inputimage.GetSpacing())
    segmentation.SetDirection(inputimage.GetDirection())
    if orgimage is not inputimage:
        segmentation = sitk.Resample(segmentation, orgimage, sitk.Transform(), sitk.sitkNearestNeighbor)
    sitk.WriteImage(segmentation, args.outfile, True)
    print('done')

    if args.paoutfile is not None:
        createParentPath(args.paoutfile)
        print('saving PA to {}...'.format(args.paoutfile), end='', flush=True)
        pa = sitk.GetImageFromArray(paarry)
        pa.SetOrigin(inputimage.GetOrigin())
        pa.SetSpacing(inputimage.GetSpacing())
        pa.SetDirection(inputimage.GetDirection())
        if orgimage is not inputimage:
            pa = sitk.Resample(pa, orgimage)
        sitk.WriteImage(pa, args.paoutfile)
        print('done')


if __name__ == '__main__':
    args = ParseArgs()
    tf.app.run(main=main, argv=[sys.argv[0]])
