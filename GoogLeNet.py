from tensorflow.keras import Model,layers,Sequential
from tensorflow.keras.layers import Conv2D,\
    BatchNormalization,Activation,AveragePooling2D,GlobalAveragePooling2D,Dense,MaxPooling2D,Flatten,Dropout
import tensorflow as tf

class Conv_BN_ReLU(Model):#将卷积层、批归一化层和激活函数层封装
    def __init__(self,filter_num,kernel_size=3,strides=1,padding='same'):
        super(Conv_BN_ReLU,self).__init__()
        self.model=Sequential([
            Conv2D(filter_num,kernel_size=kernel_size,strides=strides,padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self,x):
        x=self.model(x)
        return x

# 定义Inception模块
#---------------------------------------------------------
'''在第一个分支branch1上不做改变
   在第二个分支branch2上先经过一个1x1的卷积层，然后再经过3x3的卷积层。
   在第三个分支branch3上也要先经过一个1x1的卷积层，然后再经过5x5的卷积层。
   在第四个分支branch4上先经过一个3x3的max pooling, 然后再使用1x1的卷积层进行降维。
InceptionV1模块结构：
                                 特征拼接
           /              /                   \                  \
        1x1 conv      3x3 conv             5x5 conv        1x1 conv
          |              |                     |                  |
          |           1x1 conv             1x1 conv        3x3 max pooling
           \              \                   /                  /
                                 上一层
    四个分支，分别做卷积，然后拼接输出。
GoogleNet类
'''
#---------------------------------------------------------

class InceptionBlock(Model):
    def __init__(self,f1,f2,f3,f4,f5,f6,strides=1):
        super(InceptionBlock,self).__init__()
        self.c1=Conv_BN_ReLU(f1,kernel_size=1,strides=strides)
        self.c2_1=Conv_BN_ReLU(f2,kernel_size=1,strides=strides)
        self.c2_2=Conv_BN_ReLU(f3,kernel_size=3,strides=1)
        self.c3_1=Conv_BN_ReLU(f4,kernel_size=1, strides=strides)
        self.c3_2=Conv_BN_ReLU(f5,kernel_size=5,strides=1)
        self.p4_1=MaxPooling2D(pool_size=3,strides=1)
        self.c4_2=Conv_BN_ReLU(f6,kernel_size=3,strides=strides)

    def call(self,x):
        x1=self.c1(x)
        x2_1=self.c2_1(x)
        x2_2=self.c2_2(x2_1)
        x3_1=self.c3_1(x)
        x3_2=self.c3_2(x3_1)
        x4_1=self.p4_1(x)
        x4_2=self.c4_2(x4_1)
       #concat along axis=channel
        x=tf.concat([x1,x2_2,x3_2,x4_2],axis=3)
        return x

###############################################    model   ###############################################
class GoogLeNet(Model):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        self.aux_or_not=False
        # Stage 1
        self.c1=Conv_BN_ReLU(64, kernel_size=7, strides=2, padding="same")
        self.p1=MaxPooling2D(pool_size=3,strides=2)
        # Stage 2
        self.c2=Conv_BN_ReLU(64,kernel_size=1,strides=1)
        self.c3=Conv_BN_ReLU(192,kernel_size=3,strides=1)
        self.p2=MaxPooling2D(pool_size=3,strides=2,padding='same')
        # Stage 3
        self.inception1=InceptionBlock(64, 96, 128, 16, 32, 32,strides=1)
        self.inception2 = InceptionBlock(128, 128, 192, 32, 96, 64, strides=1)
        self.p3=MaxPooling2D(pool_size=3,strides=2,padding='same')
        # Stage 4
        self.inception3=InceptionBlock(192, 96, 208, 16, 48, 64,strides=1)
        if self.aux_or_not:
            self.aux1=InceptionAux(1000)
        self.inception4=InceptionBlock(160, 112, 224, 24, 64, 64,strides=1)
        self.inception5=InceptionBlock(128, 128, 256, 24, 64, 64,strides=1)
        self.inception6=InceptionBlock(112, 144, 288, 32, 64, 64,strides=1)
        if self.aux_or_not:
            self.aux2=InceptionAux(1000)
        self.inception7=InceptionBlock(256, 160, 320, 32, 128, 128,strides=1)
        self.p4=MaxPooling2D(pool_size=3,strides=2,padding='same')
        # Stage 5
        self.inception8=InceptionBlock(256, 160, 320, 32, 128, 128,strides=1)
        self.inception9=InceptionBlock(384, 192, 384, 48, 128, 128,strides=1)#(None, 7, 7, 1024)
        self.a1=AveragePooling2D(pool_size=7, strides=1)#(None, 1, 1, 1024)
        self.f1=Flatten()#(None,1024)
        self.d1=Dropout(0.4)
        self.dense=Dense(1000,activation='softmax')

    def call(self,x,training=None):
        x=self.c1(x)
        x=self.p1(x)

        x=self.c2(x)
        x=self.c3(x)
        x=self.p2(x)

        x=self.inception1(x)
        x=self.inception2(x)
        x=self.p3(x)

        x=self.inception3(x)
        if self.aux_or_not:
            aux1= self.aux1(x)
        x=self.inception4(x)
        x=self.inception5(x)
        x=self.inception6(x)
        if self.aux_or_not:
            aux2= self.aux2(x)
        x=self.inception7(x)
        x=self.p4(x)

        x=self.inception8(x)
        x=self.inception9(x)
        x=self.a1(x)
        x=self.f1(x)
        x=self.d1(x)
        y=self.dense(x)
        if self.aux_or_not:
            return y,aux2,aux1

        return y

class InceptionAux(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(InceptionAux, self).__init__(**kwargs)
        self.averagePool = layers.AvgPool2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, activation="relu")

        self.fc1 = layers.Dense(1024, activation="relu")
        self.fc2 = layers.Dense(num_classes)
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(inputs)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 2048
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        x = self.softmax(x)

        return x

if __name__ == '__main__':
    model=GoogLeNet()
    x=tf.random.normal([1,224,224,3])
    y=model(x)
    print(y)



#---------------------------------------------------------
'''
四个Inception结构块顺序相连，每两个Inception结构块组成
一个block，每个block中的第一个Inception结构块，卷积步长s=2，
输出特征图尺寸减半，第二个Inception结构块，卷积步长s=1
'''
#---------------------------------------------------------
# class InceptionNet10(Model):
#     def __init__(self,num_blocks,num_classes,init_filter_num=16,**kwargs):
#         super(InceptionNet10,self).__init__(**kwargs)
#         self.num_blocks=num_blocks
#         self.in_channels=init_filter_num
#         self.out_channels=init_filter_num
#         self.init_ch=init_filter_num
#         self.c1=Conv_BN_ReLU(init_filter_num)
#         self.blocks=Sequential()
#         for block_id in range(num_blocks):
#             for layer_id in range (2):
#                 if layer_id==0:
#                     block=InceptionBlock(self.out_channels,strides=2)
#                 else:
#                     block=InceptionBlock(self.out_channels,strides=1)
#                 self.blocks.add(block)
#             self.out_channels*=2
#         self.p1=GlobalAveragePooling2D()
#         self.d1=Dense(num_classes,activation='softmax')
#
#
#     def call(self,x):
#         x=self.c1(x)
#         x=self.blocks(x)
#         x=self.p1(x)
#         x=self.d1(x)
#         return x
#
# model=InceptionNet10(num_blocks=2,num_classes=10)