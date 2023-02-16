import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
import numpy as np

label_dict = {
    (  0,  0,  0): 0, 
    (  0,  0,  0): 0, 
    (  0,  0,  0): 0, 
    (  0,  0,  0): 0, 
    (  0,  0,  0): 0, 
    (111, 74,  0): 0, 
    ( 81,  0, 81): 0, 
    (128, 64,128): 1, 
    (244, 35,232): 1, 
    (250,170,160): 1, 
    (230,150,140): 1, 
    ( 70, 70, 70): 2, 
    (102,102,156): 2, 
    (190,153,153): 2, 
    (180,165,180): 2, 
    (150,100,100): 2, 
    (150,120, 90): 2, 
    (153,153,153): 3, 
    (153,153,153): 3, 
    (250,170, 30): 3, 
    (220,220,  0): 3, 
    (107,142, 35): 4, 
    (152,251,152): 4, 
    ( 70,130,180): 5, 
    (220, 20, 60): 6, 
    (255,  0,  0): 6, 
    (  0,  0,142): 7, 
    (  0,  0, 70): 7, 
    (  0, 60,100): 7, 
    (  0,  0, 90): 7, 
    (  0,  0,110): 7, 
    (  0, 80,100): 7, 
    (  0,  0,230): 7, 
    (119, 11, 32): 7, 
    (  0,  0,142): 7
}

def SE_block(inputs, reduction_factor=2):
    input_channels = int(inputs.shape[-1])
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(input_channels//reduction_factor, activation='relu')(x)
    x = layers.Dense(input_channels, activation='hard_sigmoid')(x)
    x = layers.Reshape((1, 1, input_channels))(x)
    x = layers.Multiply()([inputs, x])
    return x

class SE_Block(tf.keras.layers.Layer):
    def __init__(self, reduction_factor=4, name=None):
        super(SE_Block, self).__init__(name=name)
        self.reduction_factor = reduction_factor

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_factor': self.reduction_factor
        })
        return config

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        self.pool = layers.GlobalAveragePooling2D()
        self.d1 = layers.Dense(input_channels//self.reduction_factor, activation='relu6')
        self.d2 = layers.Dense(input_channels, activation='hard_sigmoid')
        self.reshape = layers.Reshape((1, 1, input_channels))
        self.out = layers.Multiply()

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.d1(x)
        x = self.d2(x)
        x = self.reshape(x)
        return self.out([inputs, x])

class Conv_Block(tf.keras.layers.Layer):
    def __init__(self, kernel, filters, stride, activation=tf.nn.relu6, name=None):
        super(Conv_Block, self).__init__(name=name)
        self.kernel = kernel
        self.filters = filters
        self.activation = activation
        self.stride = stride

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel': self.kernel,
            'filters': self.filters,
            'stride': self.stride,
            'activation': self.activation
        })
        return config

    def build(self, input_shape):
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.p_conv = layers.Conv2D(self.filters, 1, padding='same', activation=self.activation)
        self.d_conv = layers.DepthwiseConv2D(self.kernel, self.stride, 'same', activation=self.activation)
        # self.conv = layers.SeparableConv2D(self.filters,self.kernel,self.stride,'same',activation=self.activation)
    
    def call(self, inputs):
        x = self.p_conv(inputs)
        x = self.bn1(x)
        x = self.d_conv(x)
        x = self.bn2(x)
        return x
        # return self.conv(inputs)

# class SelfAttention_Block(tf.keras.layers.Layer):
#     # Self attention layer for n_channels
#     def __init__(self, n_channels):
#         self.n_channels = n_channels
        
#     def build(self, input_shape): 
#         n = self.n_channels
#         self.query = layers.Conv2D(n//8, 1, padding='same', use_bias=False)
#         self.key = layers.Conv2D(n//8, 1, padding='same', use_bias=False)
#         self.value = layers.Conv2D(n, 1, padding='same', use_bias=False)
#         self.gamma = tf.Variable(0, trainable=True, dtype=tf.float32)
#         self.softmax = layers.Softmax(axis=1)

#     def call(self, x):
#         size = tf.shape(x)
#         x = tf.reshape(x,[*size[:2],-1])
#         f, g, h = self.query(x), self.key(x), self.value(x)
#         beta = self.softmax(tf.matmul(f.transpose(1,2), g))
#         o = self.gamma * tf.matmul(h, beta) + x
#         return tf.reshape(o, size)        

def hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

class MNV3_Block(tf.keras.layers.Layer):
    def __init__(self, kernel, filters, stride=1, activation=tf.nn.relu6, reduction_factor=4, name=None):
        super(MNV3_Block, self).__init__(name=name)
        self.kernel = kernel
        self.filters = filters
        self.stride = stride
        self.activation = activation
        self.reduction_factor = reduction_factor

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel': self.kernel,
            'filters': self.filters,
            'stride': self.stride,
            'activation': self.activation,
            'reduction_factor': self.reduction_factor
        })
        return config

    def build(self, input_shape):
        self.conv = Conv_Block(self.kernel, self.filters, self.stride, self.activation)
        self.se = SE_Block(self.reduction_factor)
        self.out = layers.Conv2D(self.filters, 1, padding='same', activation=self.activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.se(x)
        x = self.out(x)
        return x
    
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops

def analyze(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams
    
    print('Trainable params:',trainableParams)
    print('Nontrainable params:',nonTrainableParams)
    print('Total params:', totalParams)
    print('Flops:', get_flops(model))

"""#Models"""

def get_fpn(CLASS_NUIM = 3):
    # down sample
    input = tf.keras.Input((416,416,3),name='input')
    x1 = MNV3_Block(3,16,2,hswish,name='block1')(input)
    x2 = MNV3_Block(5,40,2,name='block2')(x1)
    x3 = MNV3_Block(3,80,2,hswish,name='block3')(x2)
    x4 = MNV3_Block(3,160,2,hswish,name='block4')(x3)
    x5 = MNV3_Block(3,320,2,hswish,name='block5')(x4)
    # up sample
    p5 = layers.Conv2DTranspose(160,3,2,padding='same',name='up5')(x5)
    p5 = layers.Add(name='fuse1')([p5, layers.Conv2D(160, 1, padding='same', activation='relu6')(x4)])

    p4 = layers.Conv2DTranspose(80,3,2,padding='same',name='up4')(p5)
    p4 = layers.Add(name='fuse2')([p4, layers.Conv2D(80, 1, padding='same', activation='relu6')(x3)])

    p3 = layers.Conv2DTranspose(40,3,2,padding='same',name='up3')(p4)
    p3 = layers.Add(name='fuse3')([p3, layers.Conv2D(40, 1, padding='same', activation='relu6')(x2)])

    p2 = layers.Conv2DTranspose(16,3,2,padding='same',name='up2')(p3)
    p2 = layers.Add(name='fuse4')([p2, layers.Conv2D(16, 1, padding='same', activation='relu6')(x1)])
    p1 = layers.Conv2DTranspose(CLASS_NUIM,3,2,padding='same',name='up1')(p2)
    out = tf.keras.layers.Softmax(name='softmax_up1')(p1)

    return tf.keras.Model(inputs=[input], outputs=[out])

def get_enhanced_fpn(CLASS_NUIM = 3):
    # down sample
    input = tf.keras.Input((416,416,3),name='input')
    x1 = MNV3_Block(3,16,2,hswish,name='block1')(input)
    x2 = MNV3_Block(5,40,2,name='block2')(x1)
    x3 = MNV3_Block(3,80,2,hswish,name='block3')(x2)
    x4 = MNV3_Block(3,160,2,hswish,name='block4')(x3)
    x5 = MNV3_Block(3,320,2,hswish,name='block5')(x4)
    # up sample
    p5 = layers.Conv2DTranspose(160,3,2,padding='same',name='up5')(x5)
    p5 = layers.Add(name='fuse1')([p5, layers.Conv2D(160, 1, padding='same', activation='relu6')(x4)])

    p4 = layers.Conv2DTranspose(80,3,2,padding='same',name='up4')(p5)
    p4 = layers.Add(name='fuse2')([p4, layers.Conv2D(80, 1, padding='same', activation='relu6')(x3)])

    p3 = layers.Conv2DTranspose(40,3,2,padding='same',name='up3')(p4)
    p3 = layers.Add(name='fuse3')([p3, layers.Conv2D(40, 1, padding='same', activation='relu6')(x2)])

    p2 = layers.Conv2DTranspose(16,3,2,padding='same',name='up2')(p3)
    p2 = layers.Add(name='fuse4')([p2, layers.Conv2D(16, 1, padding='same', activation='relu6')(x1)])
    # bottom-up augmentation
    n2 = Conv_Block(3,40,2,name='bottomup1')(p2)
    n2 = layers.Add(name='fuse5')([n2, p3])

    n3 = Conv_Block(3,80,2,name='bottomup2')(n2)
    n3 = layers.Add(name='fuse6')([n3, p4])

    n4 = Conv_Block(3,160,2,name='bottomup3')(n3)
    n4 = layers.Add(name='fuse7')([n4, p5])

    n5 = Conv_Block(3,320,2,name='bottomup4')(n4)

    n5 = layers.UpSampling2D(8)(n5)
    n4 = layers.UpSampling2D(4)(n4)
    n3 = layers.UpSampling2D(2)(n3)

    out = layers.Concatenate()([n2,n3,n4,n5])
    out = layers.Conv2DTranspose(16,3,2,padding='same',name='out1')(out)
    out = layers.Conv2DTranspose(CLASS_NUIM,3,2,padding='same',name='out2')(out)
    out = tf.keras.layers.Softmax(name='softmax_out')(out)

    return tf.keras.Model(inputs=[input], outputs=[out])

def get_enhanced_efm(CLASS_NUIM = 3):
    # down sample
    input = tf.keras.Input((416,416,3),name='input')
    x1 = MNV3_Block(3,16,2,hswish,name='block1')(input)
    x2 = MNV3_Block(5,40,2,name='block2')(x1)
    x2, x2p = tf.split(x2,num_or_size_splits=2, axis=-1)
    x3 = MNV3_Block(3,80,2,hswish,name='block3')(x2)
    x3, x3p = tf.split(x3,num_or_size_splits=2, axis=-1)
    x4 = MNV3_Block(3,160,2,hswish,name='block4')(x3)
    x4, x4p = tf.split(x4,num_or_size_splits=2, axis=-1)
    x5 = MNV3_Block(3,320,2,hswish,name='block5')(x4)
    # up sample
    p5 = layers.Conv2DTranspose(160,3,2,padding='same',name='up5')(x5)
    x4 = layers.Concatenate()([x4,x4p])
    p5 = layers.Add(name='fuse1')([p5, layers.Conv2D(160, 1, padding='same', activation='relu6')(x4)])

    p4 = layers.Conv2DTranspose(80,3,2,padding='same',name='up4')(p5)
    x3 = layers.Concatenate()([x3,x3p])
    p4 = layers.Add(name='fuse2')([p4, layers.Conv2D(80, 1, padding='same', activation='relu6')(x3)])

    p3 = layers.Conv2DTranspose(40,3,2,padding='same',name='up3')(p4)
    x2 = layers.Concatenate()([x2,x2p])
    p3 = layers.Add(name='fuse3')([p3, layers.Conv2D(40, 1, padding='same', activation='relu6')(x2)])

    p2 = layers.Conv2DTranspose(16,3,2,padding='same',name='up2')(p3)
    # p2 = layers.Add(name='fuse4')([p2, layers.Conv2D(16, 1, padding='same', activation='relu6')(x1)])
    # bottom-up augmentation
    n2 = layers.SeparableConv2D(40,3,2,padding='same',name='bottomup1')(p2)
    n2 = layers.Add(name='fuse5')([n2, p3])

    n3 = layers.SeparableConv2D(80,3,2,padding='same',name='bottomup2')(n2)
    n3 = layers.Add(name='fuse6')([n3, p4])

    n4 = layers.SeparableConv2D(160,3,2,padding='same',name='bottomup3')(n3)
    n4 = layers.Add(name='fuse7')([n4, p5])

    n5 = layers.SeparableConv2D(320,3,2,padding='same',name='bottomup4')(n4)

    n5 = layers.UpSampling2D(8)(n5)
    n4 = layers.UpSampling2D(4)(n4)
    n3 = layers.UpSampling2D(2)(n3)

    out = layers.Concatenate()([n2,n3,n4,n5])
    out = layers.Conv2DTranspose(16,3,2,padding='same',name='out1')(out)
    out = layers.Conv2DTranspose(CLASS_NUIM,3,2,padding='same',name='out2')(out)
    out = tf.keras.layers.Softmax(name='softmax_out')(out)

    return tf.keras.Model(inputs=[input], outputs=[out])

def get_enhanced_efm_small(CLASS_NUIM = 3):
    # down sample
    input = tf.keras.Input((416,416,3),name='input')
    x1 = MNV3_Block(3,16,2,hswish,name='block1')(input)
    x2 = MNV3_Block(5,40,2,name='block2')(x1)
    x2, x2p = tf.split(x2,num_or_size_splits=2, axis=-1)
    x3 = MNV3_Block(3,80,2,hswish,name='block3')(x2)
    x3, x3p = tf.split(x3,num_or_size_splits=2, axis=-1)
    x4 = MNV3_Block(3,160,2,hswish,name='block4')(x3)
    x4, x4p = tf.split(x4,num_or_size_splits=2, axis=-1)
    x5 = MNV3_Block(3,320,2,hswish,name='block5')(x4)
    # up sample
    p5 = layers.Conv2DTranspose(80,3,2,padding='same',name='up5')(x5)
    p5 = layers.Add(name='fuse1')([p5, layers.Conv2D(80, 1, padding='same', activation='relu6')(x4)])
    p5 = layers.Concatenate()([p5,x4p])

    p4 = layers.Conv2DTranspose(40,3,2,padding='same',name='up4')(p5)
    p4 = layers.Add(name='fuse2')([p4, layers.Conv2D(40, 1, padding='same', activation='relu6')(x3)])
    p4 = layers.Concatenate()([p4,x3p])

    p3 = layers.Conv2DTranspose(20,3,2,padding='same',name='up3')(p4)
    p3 = layers.Add(name='fuse3')([p3, layers.Conv2D(20, 1, padding='same', activation='relu6')(x2)])
    p3 = layers.Concatenate()([p3,x2p])

    p2 = layers.Conv2DTranspose(16,3,2,padding='same',name='up2')(p3)
    # p2 = layers.Add(name='fuse4')([p2, layers.Conv2D(16, 1, padding='same', activation='relu6')(x1)])
    # bottom-up augmentation
    n2 = layers.SeparableConv2D(40,3,2,padding='same',name='bottomup1')(p2)
    n2 = layers.Add(name='fuse5')([n2, p3])

    n3 = layers.SeparableConv2D(80,3,2,padding='same',name='bottomup2')(n2)
    n3 = layers.Add(name='fuse6')([n3, p4])

    n4 = layers.SeparableConv2D(160,3,2,padding='same',name='bottomup3')(n3)
    n4 = layers.Add(name='fuse7')([n4, p5])

    n5 = layers.SeparableConv2D(320,3,2,padding='same',name='bottomup4')(n4)

    n5 = layers.UpSampling2D(8)(n5)
    n4 = layers.UpSampling2D(4)(n4)
    n3 = layers.UpSampling2D(2)(n3)

    out = layers.Concatenate()([n2,n3,n4,n5])
    out = layers.Conv2DTranspose(16,3,2,padding='same',name='out1')(out)
    out = layers.Conv2DTranspose(CLASS_NUIM,3,2,padding='same',name='out2')(out)
    out = tf.keras.layers.Softmax(name='softmax_out')(out)

    return tf.keras.Model(inputs=[input], outputs=[out])

def rgb_2_id(img):
    return np.apply_along_axis(lambda x:label_dict.get(tuple(x),0), 2, img)

def create_mask(mask_np):
    input_mask = rgb_2_id(mask_np).astype(np.int32)
    input_mask = np.eye(8,dtype=np.int32)[input_mask]

    return input_mask

def load_image(data_path):
    img_path = tf.strings.regex_replace(data_path, '_gtFine_labelIds', '_leftImg8bit')
    mask_path = data_path
    
    image = tf.io.read_file(img_path)
    input_image = tf.image.decode_png(image, channels=3)
    input_image = tf.cast(input_image, tf.float32) / 255.0
    
    mask = tf.io.read_file(mask_path)
    input_mask = tf.image.decode_png(mask)
    
    return input_image, input_mask

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels