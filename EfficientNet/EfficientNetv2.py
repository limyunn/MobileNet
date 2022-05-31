import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model,layers,Sequential
import tensorflow.keras.activations
import numpy as np
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import math
from keras_flops import get_flops
import tensorflow_addons as tfa


#################### EfficientNet V2 configs ####################

'''
     r->当前Stage中Operator重复堆叠的次数
     k->kernel_size
     s->步距stride
     e->expansion ratio
     i->input channels
     o->output channels
     c->conv_type，1代表Fused-MBConv，0代表MBConv（默认为MBConv）
     se->使用SE模块，以及se_ratio
'''
#-------------------------------------------------#

v2_base_block = [  # The baseline config for v2 models.
    'r1_k3_s1_e1_i32_o16_c1',
    'r2_k3_s2_e4_i16_o32_c1',
    'r2_k3_s2_e4_i32_o48_c1',
    'r3_k3_s2_e4_i48_o96_se0.25',
    'r5_k3_s1_e6_i96_o112_se0.25',
    'r8_k3_s2_e6_i112_o192_se0.25',
]

v2_s_block = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',
]

v2_m_block = [  # about base * (width1.6, depth2.2)
    'r3_k3_s1_e1_i24_o24_c1',
    'r5_k3_s2_e4_i24_o48_c1',
    'r5_k3_s2_e4_i48_o80_c1',
    'r7_k3_s2_e4_i80_o160_se0.25',
    'r14_k3_s1_e6_i160_o176_se0.25',
    'r18_k3_s2_e6_i176_o304_se0.25',
    'r5_k3_s1_e6_i304_o512_se0.25',
]

v2_l_block = [  # about base * (width2.0, depth3.1)
    'r4_k3_s1_e1_i32_o32_c1',
    'r7_k3_s2_e4_i32_o64_c1',
    'r7_k3_s2_e4_i64_o96_c1',
    'r10_k3_s2_e4_i96_o192_se0.25',
    'r19_k3_s1_e6_i192_o224_se0.25',
    'r25_k3_s2_e6_i224_o384_se0.25',
    'r7_k3_s1_e6_i384_o640_se0.25',
]

v2_xl_block = [  # only for 21k pretraining.
    'r4_k3_s1_e1_i32_o32_c1',
    'r8_k3_s2_e4_i32_o64_c1',
    'r8_k3_s2_e4_i64_o96_c1',
    'r16_k3_s2_e4_i96_o192_se0.25',
    'r24_k3_s1_e6_i192_o256_se0.25',
    'r32_k3_s2_e6_i256_o512_se0.25',
    'r8_k3_s1_e6_i512_o640_se0.25',
]



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



def Fused_MBConv(input_tensor,filter_num,drop_rate,expansion=4,kernel_size=(3,3),stride=1,use_se=True):
    '''
    Fusing the proj conv1x1 and depthwise_conv into a conv2d.
    [B,H,W,C] = x.shape
    '''
    in_channels=input_tensor.shape[-1]
    x = Conv2D(filter_num*expansion,kernel_size,
             strides=stride,
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


def MBConv(input_tensor,filter_num,drop_rate,expansion=6,kernel_size=(3,3),stride=1,use_se=True):
    '''
    [B,H,W,C] = x.shape
    Mobile Inverted Residual Bottleneck.
    '''
    in_channels = input_tensor.shape[-1]
    x = Conv2D(int(input_tensor.shape[-1])*expansion, (1, 1),
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
        # Stochastic depth
        if drop_rate > 0:
            x = layers.Dropout(rate=drop_rate,
                               noise_shape=(None, 1, 1, 1),  # binary dropout mask
                               )(x)
        x = layers.add([x, input_tensor])

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


def Fused_stage(inputs,filter_num,kernel_size,strides,expansion,block_num,drop_rate,use_se):

    x = Fused_MBConv(inputs, filter_num, drop_rate=drop_rate,
               expansion=expansion, kernel_size=kernel_size,
               stride=strides,use_se=use_se)

    for _ in range(1,block_num):
        # 传入参数，反复调用Fused_MBConv模块
        x = Fused_MBConv(x, filter_num,drop_rate=drop_rate,
                         kernel_size=kernel_size,expansion=expansion,
                         use_se=use_se)

    return x


def stage(inputs,filter_num,kernel_size,strides,block_num,drop_rate,expansion):

    x = MBConv(inputs, filter_num,drop_rate=drop_rate,expansion=expansion,kernel_size=kernel_size,stride=strides)

    # 重复执行MBConv模块n次
    for _ in range(1,block_num):
        # 逆残差模块
        x = MBConv(x, filter_num,drop_rate=drop_rate,kernel_size=kernel_size,expansion=expansion)

    return x



def EfficientNetV2_base(width_coefficient,
                 depth_coefficient,
                 input_shape=(224, 224, 3),
                 dropout_rate=0.2,
                 model_name=None,
                 pooling=None,
                 include_top=True,
                 cfg=None,
                 num_classes=1000,
                 **kwargs):
    '''
     v2_base_block = [  # The baseline config for v2 models.
    'r1_k3_s1_e1_i32_o16_c1',
    'r2_k3_s2_e4_i16_o32_c1',
    'r2_k3_s2_e4_i32_o48_c1',
    'r3_k3_s2_e4_i48_o96_se0.25',
    'r5_k3_s1_e6_i96_o112_se0.25',
    'r8_k3_s2_e6_i112_o192_se0.25',]
    '''

    img_input = layers.Input(shape=input_shape)

    # -------------------------------------------------#
    #   stem部分，224,224,3 -> 112,112,32
    # -------------------------------------------------#
    stem = _stem_conv(img_input, round_filters(32, width_coefficient), stride=2)

    # -------------------------------------------------#
    #   搭建EfficientNetv2主干网络
    # -------------------------------------------------#
    #  112,112,32 -> 112,112,16
    x = Fused_stage(stem,round_filters(16, width_coefficient),(3,3),strides=1,expansion=1,
                        block_num=round_repeats(1,depth_coefficient),
                        drop_rate=dropout_rate,use_se=False)

    #  112,112,16 -> 56,56,32
    x = Fused_stage(x,round_filters(32, width_coefficient),(3,3),strides=2,expansion=4,
                        block_num=round_repeats(2,depth_coefficient),
                        drop_rate=dropout_rate,use_se=False)

    #  56,56,32 -> 28,28,48
    x = Fused_stage(x, round_filters(48, width_coefficient), (3, 3), strides=2, expansion=4,
                        block_num=round_repeats(2, depth_coefficient),
                        drop_rate=dropout_rate, use_se=False)

    #  28,28,48 -> 14,14,96
    x = stage(x, round_filters(96, width_coefficient), (3, 3), strides=2, expansion=4,
                        block_num=round_repeats(3, depth_coefficient),
                        drop_rate=dropout_rate)

    #  14,14,96 -> 14,14,112
    x = stage(x, round_filters(112, width_coefficient), (3, 3), strides=1, expansion=6,
                  block_num=round_repeats(5, depth_coefficient),
                  drop_rate=dropout_rate)

    #  14,14,112 -> 7,7,192
    x = stage(x, round_filters(192, width_coefficient), (3, 3), strides=2, expansion=6,
                  block_num=round_repeats(8, depth_coefficient),
                  drop_rate=dropout_rate)

    if len(cfg)!=6:
        x = stage(x,cfg[-1][-3], (3, 3), strides=1, expansion=6,
                  block_num=round_repeats(cfg[-1][0], depth_coefficient),
                  drop_rate=dropout_rate)


    if include_top:
        # 7,7,192 ==> 7,7,1280
        x = Conv2D(round_filters(1280, width_coefficient), (1, 1), strides=1, padding='same')(x)
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

    model = Model(img_input, x,name="EfficientNetV2"+model_name)

    return model


def EfficientNetV2_S(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     classes=1000,
                     model_name='_S',
                     **kwargs):
    '''
      v2_s_block = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',]
    '''

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    cfg=[[2, 3, 1, 1, 24, 24, 0, 0],
         [4, 3, 2, 4, 24, 48, 0, 0],
         [4, 3, 2, 4, 48, 64, 0, 0],
         [6, 3, 2, 4, 64, 128, 1, 0.25],
         [9, 3, 1, 6, 128, 160, 1, 0.25],
         [15, 3, 2, 6, 160, 256, 1, 0.25]]

    return EfficientNetV2_base(1.4,1.8,
                               (380, 380, 3), dropout_rate=0.4,
                               model_name=model_name,
                               include_top=include_top, weights=weights,
                               input_tensor=input_tensor,pooling=pooling,
                               num_classes=classes,cfg=cfg,
                               **kwargs
                               )



def EfficientNetV2_M(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     classes=1000,
                     model_name='_M',
                     **kwargs):
    '''
      v2_m_block = [  # about base * (width1.6, depth2.2)
    'r3_k3_s1_e1_i24_o24_c1',
    'r5_k3_s2_e4_i24_o48_c1',
    'r5_k3_s2_e4_i48_o80_c1',
    'r7_k3_s2_e4_i80_o160_se0.25',
    'r14_k3_s1_e6_i160_o176_se0.25',
    'r18_k3_s2_e6_i176_o304_se0.25',
    'r5_k3_s1_e6_i304_o512_se0.25',]
    '''

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    cfg=[[3, 3, 1, 1, 24, 24, 0, 0],
         [5, 3, 2, 4, 24, 48, 0, 0],
         [5, 3, 2, 4, 48, 80, 0, 0],
         [7, 3, 2, 4, 80, 160, 1, 0.25],
         [14, 3, 1, 6, 160, 176, 1, 0.25],
         [18, 3, 2, 6, 176, 304, 1, 0.25],
         [5, 3, 1, 6, 304, 512, 1, 0.25]]

    return EfficientNetV2_base(1.6,2.2,
                               (456, 456, 3), dropout_rate=0.4,
                               model_name=model_name,
                               include_top=include_top, weights=weights,
                               input_tensor=input_tensor, pooling=pooling,
                               num_classes=classes,cfg=cfg,
                               **kwargs
                               )

def EfficientNetV2_L(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     classes=1000,
                     model_name='_L',
                     **kwargs):
    '''
      v2_l_block =[  # about base * (width2.0, depth3.1)
    'r4_k3_s1_e1_i32_o32_c1',
    'r7_k3_s2_e4_i32_o64_c1',
    'r7_k3_s2_e4_i64_o96_c1',
    'r10_k3_s2_e4_i96_o192_se0.25',
    'r19_k3_s1_e6_i192_o224_se0.25',
    'r25_k3_s2_e6_i224_o384_se0.25',
    'r7_k3_s1_e6_i384_o640_se0.25',]
    '''

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    cfg = [[4, 3, 1, 1, 32, 32, 0, 0],
           [7, 3, 2, 4, 32, 64, 0, 0],
           [7, 3, 2, 4, 64, 96, 0, 0],
           [10, 3, 2, 4, 96, 192, 1, 0.25],
           [19, 3, 1, 6, 192, 224, 1, 0.25],
           [25, 3, 2, 6, 224, 384, 1, 0.25],
           [7, 3, 1, 6, 384, 640, 1, 0.25]]

    return EfficientNetV2_base(2.0, 3.1,
                               (600, 600, 3), dropout_rate=0.4,
                               model_name=model_name,
                               include_top=include_top, weights=weights,
                               input_tensor=input_tensor, pooling=pooling,
                               num_classes=classes, cfg=cfg,
                               **kwargs
                               )


def EfficientNetV2_XL(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     classes=1000,
                    model_name='_XL',
                     **kwargs):
    '''
    v2_xl_block = [  # only for 21k pretraining.
    'r4_k3_s1_e1_i32_o32_c1',
    'r8_k3_s2_e4_i32_o64_c1',
    'r8_k3_s2_e4_i64_o96_c1',
    'r16_k3_s2_e4_i96_o192_se0.25',
    'r24_k3_s1_e6_i192_o256_se0.25',
    'r32_k3_s2_e6_i256_o512_se0.25',
    'r8_k3_s1_e6_i512_o640_se0.25',]
    '''

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    cfg = [ # only for 21k pretraining.
        [4, 3, 1, 1, 32, 32, 1, None],
        [8, 3, 2, 4, 32, 64, 1, None],
        [8, 3, 2, 4, 64, 96, 1, None],
        [16, 3, 2, 4, 96, 192, 0, 4],
        [24, 3, 1, 6, 192, 256, 0, 4],
        [32, 3, 2, 6, 256, 512, 0, 4],
        [8, 3, 1, 6, 512, 640, 0, 4],
    ]

    return EfficientNetV2_base(2.4, 3.6,
                               (600, 600, 3), dropout_rate=0.4,
                               model_name=model_name,
                               include_top=include_top, weights=weights,
                               input_tensor=input_tensor, pooling=pooling,
                               num_classes=classes, cfg=cfg,
                               **kwargs)


model=EfficientNetV2_L()
model.summary()
print('flops:', get_flops(model, batch_size=4))
