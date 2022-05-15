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


def _conv_block(input_tensor,filter_num,kernel_size=(3,3),stride=1):
    x=Conv2D(filter_num,kernel_size,
             strides=stride,
             padding='same',
             name='conv_1')(input_tensor)
    x=BatchNormalization(name='conv_1_bn')(x)
    x=ReLU(max_value=6.0,name='conv_1_relu')(x)
    return x

def _depthwise_conv_block(input_tensor,filter_num,kernel_size=(3,3),stride=1,block_id=1):
    x=DepthwiseConv2D((3,3),strides=stride,
                      depth_multiplier=1,
                      padding='same',
                      name='conv_dw_%d'% block_id)(input_tensor)
    x=BatchNormalization(name='conv_dw_bn_%d'% block_id)(x)
    x=ReLU(max_value=6.0,name='conv_dw_relu_%d'% block_id)(x)


    x=Conv2D(filter_num,(1,1),
             strides=1,
             padding='same',
             name='conv_pw_%d'% block_id)(x)
    x=BatchNormalization(name='conv_pw_bn_%d'% block_id)(x)
    x=ReLU(max_value=6.0,name='conv_pw_relu_%d'% block_id)(x)
    return x


def MobileNet(input_shape=[224,224,3],
              num_classes=1000,
              include_top=True):
    inputs=Input(shape=input_shape)
    # 224,224,3 -> 112,112,32
    stem=_conv_block(inputs,32,stride=2)

    # 112,112,32 -> 112,112,64
    x=_depthwise_conv_block(stem,64,block_id=1)


    # 112,112,64 -> 56,56,128
    x=_depthwise_conv_block(x,128,stride=2,block_id=2)


    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, block_id=3)

    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256,stride=2, block_id=4)


    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, block_id=5)

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512,stride=2, block_id=6)

    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, block_id=7)
    x = _depthwise_conv_block(x, 512, block_id=8)
    x = _depthwise_conv_block(x, 512, block_id=9)
    x = _depthwise_conv_block(x, 512, block_id=10)
    x = _depthwise_conv_block(x, 512, block_id=11)
    print(x.shape)


    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024,stride=2, block_id=12)
    x = _depthwise_conv_block(x, 1024, block_id=13)

    # building classifier
    if include_top is True:
         # 7,7,1024 -> 1,1,1024
         x=AveragePooling2D(pool_size=(7,7),name='avg_pool')(x)

         # 1,1,1024 -> [batch,1024]
         x=Flatten(name='flatten')(x)
         output=Dense(num_classes,activation='softmax',name='dense')(x)
    else:
        output=x

    inputs=inputs
    model = Model(inputs,output, name='mobilenetv1')
    return model


model = MobileNet()
model.summary()
plot_model(model,to_file='model_1.png',show_layer_names=True,show_shapes=True,dpi=128)

