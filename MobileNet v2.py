"""MobileNet v2 models for Keras.
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model,layers,Sequential
import numpy as np
from tensorflow.keras.utils import plot_model


def _stem_conv(input_tensor,filter_num,kernel_size=(3,3),stride=1):
    x=Conv2D(filter_num,kernel_size,
             strides=stride,
             padding='same',
             name='stem_conv')(input_tensor)
    x=BatchNormalization(name='stem_conv_bn')(x)
    x=ReLU(max_value=6.0,name='stem_conv_relu')(x)
    return x

def _bottleneck(input_tensor,filter_num,stride=1,alpha=1):
    in_channels=input_tensor.shape[-1]
    x = Conv2D(int(in_channels)*alpha, (1, 1),
               strides=1,
               padding='same'
               )(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)

    x=DepthwiseConv2D((3,3),strides=stride,
                      depth_multiplier=1,
                      padding='same'
                      )(input_tensor)
    x=BatchNormalization()(x)
    x=ReLU(max_value=6.0)(x)

    x=Conv2D(filter_num,(1,1),
             strides=1,
             padding='same',
             activation=None)(x)
    x=BatchNormalization()(x)

    if x.shape[-1]==input_tensor.shape[-1] and stride==1:
        out=Add()([x,input_tensor])

    else:
        out=x

    return out


def _inverted_residual_block(inputs,filter_num,strides,block_num,alpha):
    x=_bottleneck(inputs,filter_num,stride=strides,alpha=alpha)

    for i in range(1,block_num):
        x=_bottleneck(x,filter_num,stride=1,alpha=alpha)

    return x


def MobileNetv2(input_shape=[224,224,3],
              num_classes=1000,
              include_top=True):
    '''
     #  t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],

    '''

    inputs=Input(shape=input_shape)
    # 224,224,3 -> 112,112,32
    stem=_stem_conv(inputs,32,stride=2)

    # 112,112,32 -> 112,112,16
    x = _bottleneck(stem,16,alpha=1)

    # 112,112,16-> 56,56,24
    x = _inverted_residual_block(x, 24, strides=2, block_num=2, alpha=6)

    # 56,56,24 -> 28,28,32
    x = _inverted_residual_block(x, 32, strides=2, block_num=3, alpha=6)

    # 28,28,32 -> 14,14,64
    x = _inverted_residual_block(x, 64, strides=2, block_num=4, alpha=6)

    # 14,14,64 -> 14,14,96
    x = _inverted_residual_block(x, 96, strides=1, block_num=3, alpha=6)

    # 14,14,96 -> 7,7,160
    x = _inverted_residual_block(x, 160, strides=2, block_num=3, alpha=6)

    # 7,7,160 -> 7,7,320
    x = _inverted_residual_block(x, 320, strides=1, block_num=1, alpha=6)

    # 7,7,320 -> 7,7,1280
    x = Conv2D(1280,(1,1),strides=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)

    if include_top is True:
        # building classifier
        x = layers.GlobalAveragePooling2D()(x)  # pool + flatten
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(num_classes, name='Logits')(x)
    else:
        output = x

    inputs=inputs
    model = Model(inputs,output, name='mobilenetv2')
    return model


model = MobileNetv2()
model.summary()
plot_model(model,to_file='model_1.png',show_layer_names=True,show_shapes=True,dpi=128)

