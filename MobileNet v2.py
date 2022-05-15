"""MobileNet v2 models for Keras.
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D


from tensorflow.keras import backend as K

