import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model,layers,Sequential
import numpy as np
from tensorflow.keras.utils import plot_model


class HardSwish(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.relu6 = layers.ReLU(6.)

    def call(self, inputs, **kwargs):
        x = self.relu6(inputs + 3) * (1. / 6)
        return x

def get_activation(inputs,mode):
    if mode=='HS':
       out = HardSwish()(inputs)

    elif mode=='RE':
       out = ReLU(max_value=6.)(inputs)
    return out


def se_block(inputs,alpha=4):
    '''
    [B,H,W,C] = x.shape
    '''

    # [batch, height, width, channel] -> [batch, channel]
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(inputs.shape[-1])/alpha,activation='relu')(x)
    x = Dense(int(x.shape[-1])*alpha)(x)
    x = Activation('sigmoid')(x)
    out = Multiply()([inputs,x])
    return out


def _stem_conv(input_tensor,filter_num,kernel_size=(3,3),stride=1):
    '''
    [B,H,W,C] = x.shape
    '''
    x = Conv2D(filter_num,kernel_size,
             strides=stride,
             padding='same',
             name='stem_conv')(input_tensor)
    x = BatchNormalization(name='stem_conv_bn')(x)
    x = HardSwish(name='stem_conv_hswish')(x)
    return x

def _bottle_neck_(input_tensor,filter_num,mode,expansion,kernel_size=(3,3),stride=1,use_se=True):
    '''
        [B,H,W,C] = x.shape
        '''
    in_channels = input_tensor.shape[-1]
    x = Conv2D(expansion, (1, 1),
               strides=1,
               padding='same'
               )(input_tensor)
    x = BatchNormalization()(x)
    x = get_activation(x,mode=mode)

    x = DepthwiseConv2D(kernel_size=kernel_size, strides=stride,
                        depth_multiplier=1,
                        padding='same'
                        )(input_tensor)
    x = BatchNormalization()(x)
    x = get_activation(x,mode=mode)

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
        output = Add()([x, input_tensor])

    else:
        output = x

    return output


def MobileNetv3_Large(input_shape=[224,224,3],
              num_classes=1000,
              include_top=True):
    inputs = Input(shape=input_shape)

    # 224,224,3 -> 112,112,16
    stem = _stem_conv(inputs,16,stride=2)

    # 112,112,16 -> 112,112,16
    x = _bottle_neck_(stem, 16, 'RE', 16 , stride=1,  use_se=False)

    # 112,112,16 -> 56,56,24
    x = _bottle_neck_(x, 24, 'RE', 64 , stride=2, use_se=False)

    # 56,56,24 -> 56,56,24
    x = _bottle_neck_(x, 24, 'RE', 72 , stride=1, use_se=False)

    # 56,56,24 -> 28,28,40
    x = _bottle_neck_(x, 40, 'RE', 72  , kernel_size=(5,5), stride=2 , use_se=True)
    x = _bottle_neck_(x, 40, 'RE', 120 , kernel_size=(5,5), stride=1 , use_se=True)
    x = _bottle_neck_(x, 40, 'RE', 120 , kernel_size=(5,5), stride=1 , use_se=True)

    # 28,28,40 -> 14,14,80
    x = _bottle_neck_(x, 80, 'HS', 240 , stride=2 , use_se=False)
    x = _bottle_neck_(x, 80, 'HS', 200 , stride=1 , use_se=False)
    x = _bottle_neck_(x, 80, 'HS', 184 , stride=1 , use_se=False)
    x = _bottle_neck_(x, 80, 'HS', 184 , stride=1 , use_se=False)

    # 14,14,80 -> 14,14,112
    x = _bottle_neck_(x, 112, 'HS', 480, stride=1 , use_se=True)
    x = _bottle_neck_(x, 112, 'HS', 672, stride=1 , use_se=True)

    # 14,14,112 -> 7,7,160
    x = _bottle_neck_(x, 160, 'HS', 672, kernel_size=(5,5), stride=2 , use_se=True)
    x = _bottle_neck_(x, 160, 'HS', 960, stride=1, use_se=True)
    x = _bottle_neck_(x, 160, 'HS', 960, stride=1, use_se=True)

    # 7,7,160 -> 7,7,960
    x = Conv2D(960,(1,1),strides=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = HardSwish()(x)

    if include_top is True:

       # 7,7,960 -> 1,1,960
       x = AveragePooling2D(pool_size=(7,7),strides=1)(x)

       # 1,1,960 -> 1,1,1280
       x = Conv2D(1280,(1,1),strides=1,padding='same')(x)
       x = HardSwish()(x)

       # 1,1,1280 -> 1,1,1000
       x = Conv2D(num_classes,(1,1),strides=1, padding='same', activation='softmax')(x)

       # 1,1,1280 -> [batch,1000]
       x = Reshape((num_classes,))(x)


    model = Model(inputs, x)

    return model



def MobileNetv3_Small(input_shape=[224,224,3],
              num_classes=1000,
              include_top=True):
    inputs = Input(shape=input_shape)

    # 224,224,3 -> 112,112,16
    stem = _stem_conv(inputs, 16, stride=2)

    # 112,112,16 -> 56,56,16
    x = _bottle_neck_(stem, 16, 'RE', 16, stride=2, use_se=True)

    # 56,56,16 -> 28,28,24
    x = _bottle_neck_(x, 24, 'RE', 72, stride=2, use_se=False)

    # 28,28,24 -> 28,28,24
    x = _bottle_neck_(x, 24, 'RE', 88, stride=1, use_se=False)

    # 28,28,24 -> 14,14,40
    x = _bottle_neck_(x, 40, 'HS', 96,  kernel_size=(5,5), stride=2, use_se=True)
    x = _bottle_neck_(x, 40, 'HS', 240, kernel_size=(5,5), stride=1, use_se=True)
    x = _bottle_neck_(x, 40, 'HS', 240, kernel_size=(5,5), stride=1, use_se=True)

    # 14,14,40 -> 14,14,48
    x = _bottle_neck_(x, 48, 'HS', 120, kernel_size=(5,5), stride=1, use_se=True)
    x = _bottle_neck_(x, 48, 'HS', 144, kernel_size=(5,5), stride=1, use_se=True)

    # 14,14,48 -> 7,7,96
    x = _bottle_neck_(x, 96, 'HS', 288, kernel_size=(5,5), stride=2, use_se=True)
    x = _bottle_neck_(x, 96, 'HS', 576, kernel_size=(5,5), stride=1, use_se=True)
    x = _bottle_neck_(x, 96, 'HS', 576, kernel_size=(5,5), stride=1, use_se=True)

    # 7,7,96 -> 7,7,576
    x = Conv2D(576,(1,1),strides=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = HardSwish()(x)
    x = se_block(x)

    if include_top is True:

       # 7,7,576 -> 1,1,576
       x = AveragePooling2D(pool_size=(7,7),strides=1)(x)

       # 1,1,576 -> 1,1,1024
       x = Conv2D(1280,(1,1),strides=1,padding='same')(x)
       x = HardSwish()(x)

       # 1,1,1024 -> 1,1,1000
       x = Conv2D(num_classes,(1,1),strides=1, padding='same', activation='softmax')(x)

       model = Model(inputs, x)

       return model


model=MobileNetv3_Small()
model.summary()
# plot_model(model,to_file='model_2.png',show_layer_names=True,show_shapes=True,dpi=128)