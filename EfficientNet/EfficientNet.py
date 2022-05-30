'''
  Reference:
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
      https://arxiv.org/abs/1905.11946) (ICML 2019)

Model           |  input_size  |  width_coefficient  |  depth_coefficient  | dropout_rate
-------------------------------------------------------------------------------------------
EfficientNetB0  |   224x224    |    1.0              |      1.0            |    0.2
-------------------------------------------------------------------------------------------
EfficientNetB1  |   240x240    |    1.0              |      1.1            |    0.2
-------------------------------------------------------------------------------------------
EfficientNetB2  |   260x260    |    1.1              |      1.2            |    0.3
-------------------------------------------------------------------------------------------
EfficientNetB3  |   300x300    |    1.2              |      1.4            |    0.3
-------------------------------------------------------------------------------------------
EfficientNetB4  |   380x380    |    1.4              |      1.8            |    0.4
-------------------------------------------------------------------------------------------
EfficientNetB5  |   456x456    |    1.6              |      2.2            |    0.4
-------------------------------------------------------------------------------------------
EfficientNetB6  |   528x528    |    1.8              |      2.6            |    0.5
-------------------------------------------------------------------------------------------
EfficientNetB7  |   600x600    |    2.0              |      3.1            |    0.5

'''

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model,layers,Sequential
import numpy as np
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import math
from keras_flops import get_flops
import tensorflow_addons as tfa

params_dict = {
       #(width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-b0': (1.0, 1.0, 224, 0.2),
      'efficientnet-b1': (1.0, 1.1, 240, 0.2),
      'efficientnet-b2': (1.1, 1.2, 260, 0.3),
      'efficientnet-b3': (1.2, 1.4, 300, 0.3),
      'efficientnet-b4': (1.4, 1.8, 380, 0.4),
      'efficientnet-b5': (1.6, 2.2, 456, 0.4),
      'efficientnet-b6': (1.8, 2.6, 528, 0.5),
      'efficientnet-b7': (2.0, 3.1, 600, 0.5),
              }



#-------------------------------------------------#
#   swish activation
#-------------------------------------------------#
def swish(x, beta=1.0):
    '''
    Swish activation function: x * sigmoid(x).
    Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    '''
    return x * K.sigmoid(beta * x)


def _stem_conv(input_tensor,filter_num,kernel_size=(3,3),stride=1):
    '''
    [B,H,W,C] = x.shape
    '''
    x = Conv2D(filter_num,kernel_size,
             strides=stride,
             padding='same',
             name='stem_conv')(input_tensor)
    x = BatchNormalization(name='stem_conv_bn')(x)
    x = Activation(swish,name='stem_conv_swish')(x)
    return x



def se_block(inputs,alpha=4):
    '''
    [B,H,W,C] = x.shape
    '''
    in_channels = inputs.shape[-1]
    # [batch, height, width, channel] -> [batch, channel]
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(in_channels)/alpha,activation='relu')(x)
    x = Dense(int(in_channels))(x)
    x = Activation('sigmoid')(x)
    out = layers.multiply([inputs, x])
    return out



def MBConv(input_tensor,filter_num,drop_rate,ratio=6,kernel_size=(3,3),stride=1,use_se=True):
    '''
    [B,H,W,C] = x.shape
    Mobile Inverted Residual Bottleneck.
    '''
    in_channels = input_tensor.shape[-1]
    x = Conv2D(int(input_tensor.shape[-1])*ratio, (1, 1),
               strides=1,
               padding='same'
               )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(swish)(x)

    x = DepthwiseConv2D(kernel_size=kernel_size, strides=stride,
                        depth_multiplier=1,
                        padding='same'
                        )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(swish)(x)

    if use_se:
        out = se_block(x)

    else:
        out = x

    x = Conv2D(filter_num, (1, 1),
               strides=1,
               padding='same',
               activation=None)(out)
    x = BatchNormalization()(x)

    if x.shape[-1] == input_tensor.shape[-1] and stride == 1:
        if drop_rate > 0:

            outputs = tfa.layers.StochasticDepth(1 - drop_rate)([input_tensor, x])

        else:
            outputs = layers.add([x, input_tensor])

        return outputs


    return x


#-------------------------------------------------#
#   该函数的目的是保证filter的大小可以被8整除
#-------------------------------------------------#
def round_filters(filters,width_coefficient,divisor=8):
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


# -------------------------------------------------#
#   计算Layers的重复次数,向上取整
# -------------------------------------------------#
def round_repeats(repeats,depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))



def stage(inputs,filter_num,kernel_size,strides,block_num,drop_rate,ratio):

    x = MBConv(inputs, filter_num,drop_rate=drop_rate,ratio=ratio,kernel_size=kernel_size,stride=strides)

    # 重复执行MBConv模块n次
    for _ in range(1,block_num):
        # 逆残差模块
        x = MBConv(x, filter_num,drop_rate=drop_rate,kernel_size=kernel_size,ratio=ratio)

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 input_shape=(224, 224, 3),
                 dropout_rate=0.2,
                 model_name="efficientnet",
                 pooling=None,
                 include_top=True,
                 num_classes=1000,
                 **kwargs):
    '''
    kernel_size, repeats, in_channel, out_channel, expansion_ratio, stride, use_se
    block_args = [[3, 1, 32, 16, 1, 1, True],
                  [3, 2, 16, 24, 6, 2, True],
                  [5, 2, 24, 40, 6, 2, True],
                  [3, 3, 40, 80, 6, 2, True],
                  [5, 3, 80, 112, 6, 1, True],
                  [5, 4, 112, 192, 6, 2, True],
                  [3, 1, 192, 320, 6, 1, True]]
    '''

    img_input = layers.Input(shape=input_shape)

    # -------------------------------------------------#
    #   stem部分，224,224,3 -> 112,112,32
    # -------------------------------------------------#
    stem = _stem_conv(img_input, round_filters(32, width_coefficient), stride=2)

    # -------------------------------------------------#
    #   搭建EfficientNet主干网络
    # -------------------------------------------------#

    # 112,112,32 ==> 112,112,16
    x = stage(stem,round_filters(16, width_coefficient),
              (3,3),ratio=1,strides=1,block_num=round_repeats(1,depth_coefficient),drop_rate=dropout_rate)


    # 112,112,16 ==> 56,56,24
    x = stage(x,round_filters(24, width_coefficient),
             (3,3),ratio=6,strides=2,block_num=round_repeats(2,depth_coefficient),drop_rate=dropout_rate)


    # 56,56,24 ==> 28,28,40
    x = stage(x,round_filters(40, width_coefficient),
             (5,5),ratio=6,strides=2,block_num=round_repeats(2,depth_coefficient),drop_rate=dropout_rate)


    # 28,28,40 ==> 14,14,80
    x = stage(x, round_filters(80, width_coefficient),
              (3,3), ratio=6, strides=2, block_num=round_repeats(3,depth_coefficient), drop_rate=dropout_rate)


    # 14,14,80 ==> 14,14,112
    x = stage(x, round_filters(112, width_coefficient),
              (5,5), ratio=6, strides=1, block_num=round_repeats(3,depth_coefficient), drop_rate=dropout_rate)


    # 14,14,112 ==> 7,7,192
    x = stage(x, round_filters(192, width_coefficient),
              (5,5), ratio=6, strides=2, block_num=round_repeats(4,depth_coefficient), drop_rate=dropout_rate)

    # 7,7,192 ==> 7,7,320
    x = stage(x, round_filters(320, width_coefficient),
              (3,3), ratio=6, strides=1, block_num=round_repeats(1,depth_coefficient), drop_rate=dropout_rate)


    if include_top:
        # 7,7,320 ==> 7,7,1280
        x = Conv2D(round_filters(1280,width_coefficient),(1,1),strides=1,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(swish)(x)

        # 7,7,1280 ==> [batch,1280]
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)

        # [batch,1280]==>[batch,1000]
        x = layers.Dense(num_classes,
                         activation='softmax',
                         name='probs')(x)

    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    model = Model(img_input, x, name=model_name)

    return model



model=EfficientNet(1.0,1.0)
model.summary()

print('flops:', get_flops(model, batch_size=8))



print(len(model.layers))
plot_model(model,to_file='efficient_model.png',show_layer_names=True,show_shapes=True,dpi=128)



def EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.0, 1.0, (224,224,3), dropout_rate=0.2,
        model_name='efficientnet-b0',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, num_classes=classes,
        **kwargs
    )


def EfficientNetB1(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.0, 1.1, (240,240,3), dropout_rate=0.2,
        model_name='efficientnet-b1',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, num_classes=classes,
        **kwargs
    )


def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(
        1.1, 1.2, (260,260,3), dropout_rate=0.3,
        model_name='efficientnet-b2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, num_classes=classes,
        **kwargs
    )


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(
        1.2, 1.4, (300,300,3), dropout_rate=0.3,
        model_name='efficientnet-b3',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, num_classes=classes,
        **kwargs
    )


def EfficientNetB4(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.4, 1.8, (380,380,3), dropout_rate=0.4,
        model_name='efficientnet-b4',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, num_classes=classes,
        **kwargs
    )


def EfficientNetB5(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.6, 2.2, (456,456,3), dropout_rate=0.4,
        model_name='efficientnet-b5',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, num_classes=classes,
        **kwargs
    )


def EfficientNetB6(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.8, 2.6, (528,528,3),dropout_rate=0.5,
        model_name='efficientnet-b6',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, num_classes=classes,
        **kwargs
    )


def EfficientNetB7(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        2.0, 3.1, (600,600,3),dropout_rate=0.5,
        model_name='efficientnet-b7',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, num_classes=classes,
        **kwargs
    )


def EfficientNetL2(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        4.3, 5.3, (800,800,3), dropout_rate=0.5,
        model_name='efficientnet-l2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        pooling=pooling, num_classes=classes,
        **kwargs
    )















