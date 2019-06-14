import sys
import os
import tensorflow as tf
import numpy as np
import argparse
import SimpleITK as sitk
import random
import keras


args = None
def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("trainingdatafile", help="Input Dataset file for training")
    parser.add_argument("modelfile", help="Output trained model file in HDF5 format (*.hdf5).")
    parser.add_argument("--testfile", help="Input Dataset file for validation")
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=1000, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size", default=10, type=int)
    parser.add_argument("-l", "--learningrate", help="Learning rate", default=1e-3, type=float)
    parser.add_argument("--nobn", help="Do not use batch normalization layer", action='store_true')
    parser.add_argument("--nodropout", help="Do not use dropout layer", action='store_true')
    parser.add_argument("--noaugmentation", help="Do not use training data augmentation", action='store_true')
    parser.add_argument("--magnification", help="Magnification coefficient for data augmentation", default=4, type=int)
    parser.add_argument("--latestfile", help="The filename of the latest weights.")
    parser.add_argument("--bestfile", help="The filename of the best weights.")
    parser.add_argument("--weightinterval", help="The interval between checkpoint for weight saving.", type=int)
    parser.add_argument("--weightfile", help="The filename of the trained weight parameters file for fine tuning or resuming.")
    parser.add_argument("--premodel", help="The filename of the previously trained model")
    parser.add_argument("--initialepoch", help="Epoch at which to start training for resuming a previous training", default=0, type=int)
    #parser.add_argument("--idlist", help="The filename of ID list for splitting input datasets into training and validation datasets.")
    #parser.add_argument("--split", help="Fraction of the training data to be used as validation data.", default=0.0, type=float)
    parser.add_argument("--logdir", help="Log directory", default='log')
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args

def import_img():
    pass



def createParentPath(filepath):
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)


def ReadSliceDataList(filename):
    datalist = []
    with open(filename) as f:
        for line in f:
            imagefile, labelfile = line.strip().split(' ')
            datalist.append((imagefile, labelfile))
    return datalist


def ImportImage(filename):
    image = sitk.ReadImage(filename)
    imagearray = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() == 1:
        imagearray = imagearray[..., np.newaxis]
    return imagearray


def GetInputShapes(filenamepair):
    image = ImportImage(filenamepair[0])
    label = ImportImage(filenamepair[1])
    return (image.shape, label.shape)


def GetMinimumValue(image):
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(image)
    return minmax.GetMinimum()


def Affine(t, r, scale, shear, c):
    a = sitk.AffineTransform(2)
    a.SetCenter(c)
    a.Scale(scale)
    a.Rotate(0,1,r)
    a.Shear(0,1,shear[0])
    a.Shear(1,0,shear[1])
    a.Translate(t)
    return a



def Transforming(image, bspline, affine, interpolator, minval):
    # B-spline transformation
    transformed_b = sitk.Resample(image, bspline, interpolator, minval)
    # Affine transformation
    transformed_a = sitk.Resample(transformed_b, affine, interpolator, minval)

    return transformed_a

#画像のTransformed
def ImportImageTransformed(imagefile, labelfile):
    sigma = 4
    translationrange = 5 # [mm]
    rotrange = 5 # [deg]
    shearrange = 1/16 
    scalerange = 0.05

    image = sitk.ReadImage(imagefile)
    label = sitk.ReadImage(labelfile)

    # B-spline parameters
    bspline = sitk.BSplineTransformInitializer(image, [5,5])
    p = bspline.GetParameters()
    numbsplineparams = len(p)
    coeff = np.random.normal(0, sigma, numbsplineparams)
    bspline.SetParameters(coeff)

    # Affine parameters
    translation = np.random.uniform(-translationrange, translationrange, 2)
    rotation = np.radians(np.random.uniform(-rotrange, rotrange))
    shear = np.random.uniform(-shearrange, shearrange, 2)
    scale = np.random.uniform(1-scalerange, 1+scalerange)
    center = np.array(image.GetSize()) * np.array(image.GetSpacing()) / 2
    affine = Affine(translation, rotation, scale, shear, center)
    minval = GetMinimumValue(image)

    transformed_image = Transforming(image, bspline, affine, sitk.sitkLinear, minval)
    transformed_label = Transforming(label, bspline, affine, sitk.sitkNearestNeighbor, 0)

    imagearry = sitk.GetArrayFromImage(transformed_image)
    imagearry = imagearry[..., np.newaxis]
    labelarry = sitk.GetArrayFromImage(transformed_label)

    return imagearry, labelarry

#バッチサイズに変換して渡している。
def ImportBatchArray(datalist, batch_size = 32, apply_augmentation = False):
    while True:
        indices = list(range(len(datalist)))
        random.shuffle(indices)

        if apply_augmentation:
            for i in range(0, len(indices), batch_size):
                #AugmentationするならImportImageTransformedを使う
                imagelabellist = [ ImportImageTransformed(datalist[idx][0], datalist[idx][1]) for idx in indices[i:i+batch_size] ]
                print("apply_augmentation")
                imagelist, onehotlabellist = zip(*imagelabellist)
                print("patch shape1 :",imagelist.shape)
                yield (np.array(imagelist), np.array(onehotlabellist))
        else:
            for i in range(0, len(indices), batch_size):
                #そうでないならImportImageを使用する。
                imagelist = np.array([ ImportImage(datalist[idx][0]) for idx in indices[i:i+batch_size] ])
                
                onehotlabellist = np.array([ keras.utils.to_categorical(ImportImage(datalist[idx][1]),num_classes=3) for idx in indices[i:i+batch_size] ])
                
                yield (imagelist, onehotlabellist)


def CreateConvBlock(x, filters, n = 2, use_bn = True, apply_pooling = True, name = 'convblock'):
    for i in range(1,n+1):
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', name=name+'_conv'+str(i))(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=name+'_BN'+str(i))(x)
        x = tf.keras.layers.Activation('relu', name=name+'_relu'+str(i))(x)

    convresult = x

    if apply_pooling:
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2), name=name+'_pooling')(x)

    return x, convresult

# def CreateConvBlockwithLSTM(x, filters, n = 2, use_bn = True, apply_pooling = True, name = 'convblock'):
#     x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', name=name+'_conv'+str(i))(x)
#     if use_bn:
#         x = tf.keras.layers.BatchNormalization(name=name+'_BN'+str(i))(x)
#     x = tf.keras.layers.Activation('relu', name=name+'_relu'+str(i))(x)

#     x = tf.keras.layers.Reshape()

#     convresult = x

#     if apply_pooling:
#         x = tf.keras.layers.MaxPool2D(pool_size=(2,2), name=name+'_pooling')(x)

#     return x, convresult


def CreateUpConvBlock(x, contractpart, filters, n = 2, use_bn = True, name = 'upconvblock'):
    # upconv x
    #unsupported operand type(s) for /: 'Dimension' and 'float'のためint追加
    x = tf.keras.layers.Conv2DTranspose(int(x.shape[-1]), (2,2), strides=(2,2), padding='same', name=name+'_upconv')(x)
    # concatenate contract4 and x
    x = tf.keras.layers.concatenate([contractpart, x])
    # conv x 2 times
    for i in range(1,n+1):
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', name=name+'_conv'+str(i))(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=name+'_BN'+str(i))(x)
        x = tf.keras.layers.Activation('relu', name=name+'_relu'+str(i))(x)

    return x


def ConstructModel(input_images, nclasses, use_bn = True, use_dropout = True):

    # Contract1 (256->128)
    with tf.name_scope("contract1"):
        x, contract1 = CreateConvBlock(input_images, 64, n = 2, use_bn = use_bn, name = 'contract1')

    # Contract2 (128->64)
    with tf.name_scope("contract2"):
        x, contract2 = CreateConvBlock(x, 128, n = 2, use_bn = use_bn, name = 'contract2')

    # Contract3 (64->32)
    with tf.name_scope("contract3"):
        x, contract3 = CreateConvBlock(x, 256, n = 2, use_bn = use_bn, name = 'contract3')

    # Contract4 (32->16)
    with tf.name_scope("contract4"):
        x, contract4 = CreateConvBlock(x, 512, n = 2, use_bn = use_bn, name = 'contract4')

    # Contract5 (16)
    with tf.name_scope("contract5"):
        x, _ = CreateConvBlock(x, 1024, n = 2, use_bn = use_bn, apply_pooling = False, name = 'contract5')

    # Dropout (16)
    with tf.name_scope("dropout"):
        if use_dropout:
            x = tf.keras.layers.Dropout(0.5, name='dropout')(x)

    # Expand4 (16->32)
    with tf.name_scope("expand4"):
        x = CreateUpConvBlock(x, contract4, 512, n = 2, use_bn = use_bn, name = 'expand4')

    # Expand3 (32->64)
    with tf.name_scope("expand3"):
        x = CreateUpConvBlock(x, contract3, 256, n = 2, use_bn = use_bn, name = 'expand3')

    # Expand2 (64->128)
    with tf.name_scope("expand2"):
        x = CreateUpConvBlock(x, contract2, 128, n = 2, use_bn = use_bn, name = 'expand2')

    # Expand1 (128->256)
    with tf.name_scope("expand1"):
        x = CreateUpConvBlock(x, contract1, 64, n = 2, use_bn = use_bn, name = 'expand1')

    # Segmentation
    with tf.name_scope("segmentation"):
        layername = 'segmentation'
        if nclasses == 3: # 8 organs segmentation
            layername = 'segmentation'
        else:
            layername = 'segmentation_{}classes'.format(nclasses)
        x = tf.keras.layers.Conv2D(nclasses, (1,1), activation='softmax', padding='same', name=layername)(x)
        #x = tf.keras.layers.Conv2D(nclasses, (1,1), activation='softmax', padding='same', name='segmentation')(x)

    return x


def dice(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice


class LatestWeightSaver(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename_ = filename

    def on_epoch_end(self, epoch, logs):
        self.model.save_weights(self.filename_)


class PeriodicWeightSaver(tf.keras.callbacks.Callback):
    def __init__(self, logdir, interval):
        self.logdir_ = logdir
        self.interval_ = interval

    def on_epoch_end(self, epoch, logs):
        if epoch % self.interval_ == 0:
            filename = self.logdir_ + "/weights_e{:02d}.hdf5".format(epoch)
            self.model.save_weights(filename)


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    trainingdatalist = ReadSliceDataList(args.trainingdatafile)
    testdatalist = None
    if args.testfile is not None:
        testdatalist = ReadSliceDataList(args.testfile)
        testdatalist = random.sample(testdatalist, int(len(testdatalist)*0.1))

    (imageshape, labelshape) = GetInputShapes(trainingdatalist[0])
    nclasses = 3 # Number of classes
    print(imageshape)
    print(labelshape)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        x = tf.keras.layers.Input(shape=imageshape, name="x")
        segmentation = ConstructModel(x, nclasses, not args.nobn, not args.nodropout)
        model = tf.keras.models.Model(x, segmentation)
        model.summary()

        optimizer = tf.keras.optimizers.Adam(lr=args.learningrate)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[dice])

    createParentPath(args.modelfile)
    with open(args.modelfile, 'w') as f:
        f.write(model.to_yaml())

    if args.weightfile is None:
        initial_epoch = 0
    else:
        model.load_weights(args.weightfile)
        initial_epoch = args.initialepoch

    if args.latestfile is None:
        latestfile = args.logdir + '/latestweights.hdf5'
    else:
        latestfile = args.latestfile
        createParentPath(latestfile)

    tb_cbk = tf.keras.callbacks.TensorBoard(log_dir=args.logdir)
    latest_cbk = LatestWeightSaver(latestfile)
    callbacks = [tb_cbk, latest_cbk]
    if testdatalist is not None:
        if args.bestfile is None:
            bestfile = args.logdir + '/bestweights.hdf5'
        else:
            bestfile = args.bestfile
            createParentPath(bestfile)
        chkp_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=bestfile, save_best_only = True, save_weights_only = True)
        callbacks.append(chkp_cbk)
    if args.weightinterval is not None:
        periodic_cbk = PeriodicWeightSaver(logdir=args.logdir, interval=args.weightinterval)
        callbacks.append(periodic_cbk)

    steps_per_epoch = len(trainingdatalist) // args.batchsize 
    print ("Batch size: {}".format(args.batchsize))
    print ("Number of Epochs: {}".format(args.epochs))
    print ("Number of Steps/epoch: {}".format(steps_per_epoch))

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        if testdatalist is not None:
            #trainingdatalist->たぶんImagepath
            model.fit_generator(ImportBatchArray(trainingdatalist, batch_size = args.batchsize, apply_augmentation = False),
                    steps_per_epoch = steps_per_epoch, epochs = args.epochs,
                    callbacks=callbacks,
                    validation_data = ImportBatchArray(testdatalist, batch_size = args.batchsize),
                    validation_steps = len(testdatalist),
                    initial_epoch = initial_epoch)
        else:
            model.fit_generator(ImportBatchArray(trainingdatalist, batch_size = args.batchsize, apply_augmentation = False),
                    steps_per_epoch = steps_per_epoch, epochs = args.epochs,
                    callbacks=callbacks,
                    initial_epoch = initial_epoch)


if __name__ == '__main__':
    args = ParseArgs()
    tf.app.run(main=main, argv=[sys.argv[0]])
