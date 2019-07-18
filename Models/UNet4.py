from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers import Input
from keras.models import Model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
import tensorflow as tf
from Utils.status import ON_WIN
from tensorflow.python.keras.utils import multi_gpu_model, plot_model


class UNet4:
    with tf.device("/cpu:0"):
        def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count,im_size, parser):
            self.name = self.__class__.__name__.lower()
            self.INPUT_IMAGE_SIZE = im_size
            self.CONCATENATE_AXIS = -1
            self.CONV_FILTER_SIZE = 4
            self.CONV_STRIDE = 2
            self.CONV_PADDING = (1, 1)
            self.DECONV_FILTER_SIZE = 2
            self.DECONV_STRIDE = 2
            self.parser = parser
            # (128 x 128 x input_channel_count)
            inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

            # エンコーダーの作成
            #↓first_filterはフィルターの数。
            # (128 x 128 x N)
            enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
            enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

            # (64 x 64 x 2N)
            filter_count = first_layer_filter_count * 2
            enc2 = self._add_encoding_layer(filter_count, enc1)

            # (32 x 32 x 4N)
            filter_count = first_layer_filter_count * 4
            enc3 = self._add_encoding_layer(filter_count, enc2)

            # (16 x 16 x 8N)
            filter_count = first_layer_filter_count * 8
            enc4 = self._add_encoding_layer(filter_count, enc3)


            # デコーダーの作成
            # (2 x 2 x 8N)
            dec1 = self._add_decoding_layer(filter_count, True, enc4)
            dec1 = concatenate([dec1, enc3], axis=self.CONCATENATE_AXIS)

            # (4 x 4 x 8N)
            filter_count = first_layer_filter_count * 4
            dec2 = self._add_decoding_layer(filter_count, True, dec1)
            dec2 = concatenate([dec2, enc2], axis=self.CONCATENATE_AXIS)

            # (8 x 8 x 8N)
            filter_count = first_layer_filter_count * 2
            dec3 = self._add_decoding_layer(filter_count, True, dec2)
            dec3 = concatenate([dec3, enc1], axis=self.CONCATENATE_AXIS)


            # (256 x 256 x output_channel_count)
            dec4 = Activation(activation='relu')(dec3)
            dec4 = Conv2DTranspose(
                output_channel_count,
                self.DECONV_FILTER_SIZE,
                activation='softmax',
                strides=self.DECONV_STRIDE)(dec4)

            model=Model(inputs, dec4)
            self.UNET = model

    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        #he_uniform
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='glorot_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self):
        return self.UNET

    def get_name(self):
        return self.name
