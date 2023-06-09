import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
import numpy as np

label_names = {
    'void': 0, 
    'flat': 1, 
    'construction': 2, 
    'object': 3, 
    'nature': 4, 
    'sky': 5, 
    'human': 6, 
    'vehicle': 7,
}

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

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
    emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb

class PositionalEncoding1D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.
        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".
        """
        super(PositionalEncoding1D, self).__init__()

        self.channels = int(np.ceil(channels / 2) * 2)
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )
        self.cached_penc = None

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(inputs.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, org_channels = inputs.shape

        dtype = self.inv_freq.dtype
        pos_x = tf.range(x, dtype=dtype)
        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        emb = tf.expand_dims(get_emb(sin_inp_x), 0)
        emb = emb[0]  # A bit of a hack
        self.cached_penc = tf.repeat(
            emb[None, :, :org_channels], tf.shape(inputs)[0], axis=0
        )

        return self.cached_penc 
    
class PositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.
        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".
        """
        super(PositionalEncoding2D, self).__init__()

        self.channels = int(2 * np.ceil(channels / 4))
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )
        self.cached_penc = None

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(inputs.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, y, org_channels = inputs.shape

        dtype = self.inv_freq.dtype

        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)

        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = tf.expand_dims(get_emb(sin_inp_x), 1)
        emb_y = tf.expand_dims(get_emb(sin_inp_y), 0)

        emb_x = tf.tile(emb_x, (1, y, 1))
        emb_y = tf.tile(emb_y, (x, 1, 1))
        emb = tf.concat((emb_x, emb_y), -1)
        self.cached_penc = tf.repeat(
            emb[None, :, :, :org_channels], tf.shape(inputs)[0], axis=0
        )
        return self.cached_penc

def SE_block(inputs, reduction_factor=2):
    input_channels = int(inputs.shape[-1])
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(input_channels//reduction_factor, activation='relu')(x)
    x = layers.Dense(input_channels, activation='hard_sigmoid')(x)
    x = layers.Reshape((1, 1, input_channels))(x)
    x = layers.Multiply()([inputs, x])
    return x

@tf.keras.utils.register_keras_serializable()
class SE_Block(tf.keras.layers.Layer):
    def __init__(self, reduction_factor=4, **kwargs):
        super(SE_Block, self).__init__(**kwargs)
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

@tf.keras.utils.register_keras_serializable()
class Conv_Block(tf.keras.layers.Layer):
    def __init__(self, kernel, filters, stride, activation=tf.nn.relu6, **kwargs):
        super(Conv_Block, self).__init__(**kwargs)
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
        # self.bn2 = layers.BatchNormalization()
        # self.p_conv = layers.Conv2D(self.filters, 1, padding='same', activation=self.activation)
        # self.d_conv = layers.DepthwiseConv2D(self.kernel, self.stride, 'same', activation=self.activation)
        self.conv = layers.SeparableConv2D(self.filters,self.kernel,self.stride,'same',activation=self.activation)
    
    def call(self, inputs):
        # x = self.p_conv(inputs)
        # x = self.bn1(x)
        # x = self.d_conv(x)
        # x = self.bn2(x)
        # return x
        x = self.conv(inputs)
        x = self.bn1(x)
        return x

@tf.keras.utils.register_keras_serializable()
class SelfAttention_Block(tf.keras.layers.Layer):
    def __init__(self, reduction_factor, **kwargs):
        super(SelfAttention_Block, self).__init__(**kwargs)
        self.reduction_factor = reduction_factor
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_factor': self.reduction_factor
        })
        return config
        
    # input shape requires channel last
    def build(self, input_shape): 
        n = input_shape[-1]
        r = self.reduction_factor
        self.query = layers.Conv2D(n//r, 1, padding='same', use_bias=False)
        self.key = layers.Conv2D(n//r, 1, padding='same', use_bias=False)
        self.value = layers.Conv2D(n, 1, padding='same', use_bias=False)
        self.reshape1 = tf.keras.layers.Reshape((-1, n//r))
        self.reshape2 = tf.keras.layers.Reshape((-1, n))
        self.gamma = tf.Variable(0, trainable=True, dtype=tf.float32)
        self.softmax = layers.Softmax(axis=1)

    def call(self, x):
        size = tf.shape(x)
        f, g, h = self.query(x), self.key(x), self.value(x)
        f, g, h = self.reshape1(f), self.reshape1(g), self.reshape2(h)
        beta = self.softmax(tf.matmul(f, tf.transpose(g, perm=[0,2,1])))
        beta = tf.reshape(tf.matmul(h, beta, transpose_a=True), size) 
        return self.gamma * beta + x       

@tf.keras.utils.register_keras_serializable()
class DualSelfAttention_Block(tf.keras.layers.Layer):
    def __init__(self, reduction_factor=1, identity=False, **kwargs):
        super(DualSelfAttention_Block, self).__init__(**kwargs)
        self.reduction_factor = reduction_factor
        self.identity = identity
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_factor': self.reduction_factor,
            'identity': self.identity
        })
        return config
        
    # input shape requires channel last
    def build(self, input_shape): 
        n = input_shape[-1]
        r = self.reduction_factor
        self.p_query =  layers.Conv2D(n//r, 1, padding='same', use_bias=False, activation='relu')
        self.p_key   =  layers.Conv2D(n//r, 1, padding='same', use_bias=False, activation='relu')
        self.p_value =  layers.Conv2D(n, 1, padding='same', use_bias=False, activation='relu')
        self.c_conv  =  layers.Conv2D(n, 1, padding='same', use_bias=False, activation='relu')
        self.reshape1 = tf.keras.layers.Reshape((-1, n//r))
        self.reshape2 = tf.keras.layers.Reshape((-1, n))
        self.position_attention =   layers.Attention(use_scale=True)
        self.channel_attention  =   layers.Attention(use_scale=True)
        self.fuse = layers.Add()

    def call(self, x):
        q, k, v = self.p_query(x), self.p_key(x), self.p_value(x)
        q, k, v = self.reshape1(q), self.reshape1(k), self.reshape2(v)
        pa = self.position_attention([q, v, k])
        pa = tf.reshape(pa, tf.shape(x)) 
        xt = tf.transpose(self.c_conv(x), [0, 3, 1, 2])
        ca = self.channel_attention([xt, xt, xt])
        ca = tf.transpose(ca, [0, 2, 3, 1])
        if self.identity == True:
            return self.fuse([x, pa, ca])
        else:
            return self.fuse([pa, ca])

@tf.keras.utils.register_keras_serializable()
def hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

@tf.keras.utils.register_keras_serializable()
class MNV3_Block(tf.keras.layers.Layer):
    def __init__(self, kernel, filters, stride=1, activation=tf.nn.relu6, reduction_factor=4, **kwargs):
        super(MNV3_Block, self).__init__(**kwargs)
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

# mobilenet self attantion, replace se block with self attantion block
@tf.keras.utils.register_keras_serializable()
class MNSA_Block(tf.keras.layers.Layer):
    def __init__(self, kernel, filters, stride=1, activation=tf.nn.relu6, reduction_factor=1, dual_attention=True, **kwargs):
        super(MNSA_Block, self).__init__(**kwargs)
        self.kernel = kernel
        self.filters = filters
        self.stride = stride
        self.activation = activation
        self.reduction_factor = reduction_factor
        self.dual_attention = dual_attention

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel': self.kernel,
            'filters': self.filters,
            'stride': self.stride,
            'activation': self.activation,
            'reduction_factor': self.reduction_factor,
            'dual_attention': self.dual_attention
        })
        return config

    def build(self, input_shape):
        self.conv = Conv_Block(self.kernel, self.filters, self.stride, self.activation)
        if (self.dual_attention):
            self.sa = DualSelfAttention_Block(self.reduction_factor, identity=True)
        else:
            self.sa = SelfAttention_Block(self.reduction_factor)
        self.out = layers.Conv2D(self.filters, 1, padding='same', activation=self.activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.sa(x)
        x = self.out(x)
        return x
    
@tf.keras.utils.register_keras_serializable()
class MHSA_Block(tf.keras.layers.Layer):
    def __init__(self, num_heads=1, **kwargs):
        super(MHSA_Block, self).__init__(**kwargs)
        self.num_heads = num_heads
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads
        })
        return config
        
    # input shape requires channel last
    def build(self, input_shape): 
        n = input_shape[-1]
        self.query =  layers.Conv2D(n, 1, padding='same', use_bias=False, activation='relu')
        self.key   =  layers.Conv2D(n, 1, padding='same', use_bias=False, activation='relu')
        self.value =  layers.Conv2D(n, 1, padding='same', use_bias=False, activation='relu')
        self.attention = layers.MultiHeadAttention(self.num_heads, n)
        self.encoding = PositionalEncoding2D(n)

    def call(self, x):
        x = self.encoding(x)
        q, k, v = self.query(x), self.key(x), self.value(x)
        out = self.attention(q, v, k)
        return out

@tf.keras.utils.register_keras_serializable()
class MHCA_Block(tf.keras.layers.Layer):
    def __init__(self, num_heads=1, **kwargs):
        super(MHCA_Block, self).__init__(**kwargs)
        self.num_heads = num_heads
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads
        })
        return config
        
    # input shape requires channel last
    def build(self, input_shape): 
        n1 = input_shape[0][-1]
        n2 = input_shape[1][-1]
        self.query =  layers.Conv2D(n2, 1, padding='same', use_bias=False, activation='relu')
        self.key   =  layers.Conv2D(n2, 1, padding='same', use_bias=False, activation='relu')
        self.value =  layers.Conv2D(n1, 3, strides=2, padding='same', use_bias=False, activation='relu')
        self.conv1 =  layers.Conv2D(n1, 1, padding='same', use_bias=False)
        self.conv2 =  layers.Conv2D(n2, 3, padding='same', use_bias=False, activation='relu')
        self.conv3 =  layers.Conv2D(n2//2, 1, padding='same', use_bias=False, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.sigmoid = layers.Activation('sigmoid')
        self.up1 = layers.UpSampling2D()
        self.up2 = layers.UpSampling2D()
        self.multiply = layers.Multiply()
        self.concat = layers.Concatenate()
        self.attention = layers.MultiHeadAttention(self.num_heads, n1)
        self.encoding1 = PositionalEncoding2D(n1)
        self.encoding2 = PositionalEncoding2D(n2)

    # expect a list of 2 inputs [x1, x2], where x1 with shape(b, 2w, 2h, d) and x2 with shape(b, w, h, 2d)
    def call(self, x):
        x1, x2 = x
        x1 = self.encoding1(x1)
        x2 = self.encoding2(x2)
        q, k, v = self.query(x2), self.key(x2), self.value(x1)
        atten = self.attention(q, v, k)
        atten = self.conv1(atten)
        atten = self.bn1(atten)
        atten = self.sigmoid(atten)
        atten = self.up1(atten)
        x1 = self.multiply([atten, x1])
        x2 = self.up2(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.bn2(x2)
        out = self.concat([x1,x2])
        return out

@tf.keras.utils.register_keras_serializable()
class CCA_Block(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CCA_Block, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        config.update({
        })
        return config
        
    # input shape requires channel last
    def build(self, input_shape): 
        n1 = input_shape[0][-1]
        n2 = input_shape[1][-1]
        assert n1 == n2
        self.query =  layers.Conv2D(n1, 1, padding='same', use_bias=False, activation='relu')
        self.key   =  layers.Conv2D(n1, 1, padding='same', use_bias=False, activation='relu')
        self.value =  layers.Conv2D(n1, 1, padding='same', use_bias=False, activation='relu')
        self.sigmoid = layers.Conv2D(n1, 1, padding='same', activation='sigmoid')
        self.multiply = layers.Multiply()
        self.bn = layers.BatchNormalization()
        self.add = layers.Add()
        self.attention = layers.Attention(use_scale=True)
        # self.attention = layers.MultiHeadAttention(1,input_shape[1][-2]*input_shape[1][-3], attention_axes=1)

    # expect a list of 2 inputs [x1, x2], where x1 with shape(b, w, h, d) and x2 with shape(b, w, h, d)
    def call(self, x):
        x1, x2 = x
        q, k, v = self.query(x1), self.key(x1), self.value(x2)
        q = tf.transpose(q, [0, 3, 1, 2])
        k = tf.transpose(k, [0, 3, 1, 2])
        v = tf.transpose(v, [0, 3, 1, 2])
        atten = self.attention([q, v, k])
        # atten = self.attention(q, v, k)
        atten = tf.transpose(atten, [0, 2, 3, 1])
        atten = self.sigmoid(atten)
        atten = self.bn(atten)
        x2a = self.multiply([x2, atten])
        out = self.add([x1, x2a])
        return out

@tf.keras.utils.register_keras_serializable()
class Conv3_Block(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(Conv3_Block, self).__init__(**kwargs)
        self.filters = filters
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters
        })
        return config
        
    def build(self, input_shape): 
        n = self.filters
        self.conv1 = layers.Conv2D(n, 3, padding='same')
        self.conv2 = layers.Conv2D(n, 3, padding='same')
        self.conv3 = layers.Conv2D(n, 3, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        out = self.relu(x)
        return out

@tf.keras.utils.register_keras_serializable()
class UnetDown_Block(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(UnetDown_Block, self).__init__(**kwargs)
        self.filters = filters
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters
        })
        return config
        
    # input shape requires channel last
    def build(self, input_shape): 
        n = self.filters
        self.conv1 = Conv3_Block(n)
        self.conv2 = Conv3_Block(n)
        self.conv3 = Conv3_Block(n)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out

@tf.keras.utils.register_keras_serializable()
class VOV_Block(tf.keras.layers.Layer):
    def __init__(self, filters_out, n_layers, layer_filter=None, activation='relu6', reduction_factor=4, identity=False, use_mobile=False, **kwargs):
        super(VOV_Block, self).__init__(**kwargs)
        self.filters_out = filters_out
        self.n_layers = n_layers
        self.layer_filter = layer_filter
        self.activation = activation
        self.reduction_factor = reduction_factor
        self.identity = identity
        self.use_mobile = use_mobile
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters_out': self.filters_out,
            'layer_filter': self.layer_filter,
            'n_layers': self.n_layers,
            'activation': self.activation,
            'reduction_factor': self.reduction_factor,
            'identity': self.identity,
            'use_mobile': self.use_mobile,
        })
        return config

    # input shape requires channel last
    def build(self, input_shape): 
        n  = self.layer_filter if self.layer_filter is not None else input_shape[-1]
        if (self.use_mobile):
            self.convs = [Conv_Block(3, n, 1, activation=self.activation) for _ in range(self.n_layers)]
        else:
            self.convs = [layers.Conv2D(n, 3, padding='same', activation=self.activation) for _ in range(self.n_layers)]
        self.conv_out = layers.Conv2D(self.filters_out, 1, padding='same', activation=self.activation)
        self.se = SE_Block(self.reduction_factor)
        if (self.identity):
            self.add = layers.Add()

    def call(self, inputs):
        x = inputs
        for c in self.convs:
            x = c(x)
        out = self.conv_out(x)
        out = self.se(out)
        if (self.identity):
            out = self.add([inputs, out])
        return out

@tf.keras.utils.register_keras_serializable()
class ELAN_Block(tf.keras.layers.Layer):
    def __init__(self, filters_out, n_layers, layer_filter=None, activation='relu6', reduction_factor=4, identity=False, use_mobile=False, **kwargs):
        super(ELAN_Block, self).__init__(**kwargs)
        self.filters_out = filters_out
        self.n_layers = n_layers
        self.layer_filter = layer_filter
        self.activation = activation
        self.reduction_factor = reduction_factor
        self.identity = identity
        self.use_mobile = use_mobile
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters_out': self.filters_out,
            'layer_filter': self.layer_filter,
            'n_layers': self.n_layers,
            'activation': self.activation,
            'reduction_factor': self.reduction_factor,
            'identity': self.identity,
            'use_mobile': self.use_mobile,
        })
        return config

    # input shape requires channel last
    def build(self, input_shape): 
        n = self.filters_out//2
        self.vov = VOV_Block(n, self.n_layers, self.layer_filter, self.activation, self.reduction_factor, self.identity, self.use_mobile)
        self.concat = layers.Concatenate()
        if (self.use_mobile):
            self.conv = Conv_Block(3, n, 1, activation=self.activation)
        else:
            self.conv = layers.Conv2D(n, 3, padding='same', activation=self.activation)

    def call(self, inputs):
        x1,x2 = tf.split(inputs,num_or_size_splits=2, axis=-1)
        x1 = self.vov(x1)
        x2 = self.conv(x2)
        out = self.concat([x1,x2])

        return out

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
    print('='*21, ' Model Info', '='*21)
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams
    
    print('Trainable params:',trainableParams)
    print('Nontrainable params:',nonTrainableParams)
    print('Total params:', totalParams)
    print('Flops:', get_flops(model))
    print('='*21, 'End of Info', '='*21)

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


def get_eefm_attention(CLASS_NUM = 3):
    # down sample with CSP
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
    x4 = layers.Conv2D(160, 1, padding='same', activation='relu6')(x4)
    x4a = layers.Attention(use_scale=True)([x4,x4,x4])
    # x4 = SelfAttention_Block(4)(x4)
    p5 = layers.Add(name='fuse1')([p5, x4, x4a])

    p4 = layers.Conv2DTranspose(80,3,2,padding='same',name='up4')(p5)
    x3 = layers.Concatenate()([x3,x3p])
    x3 = layers.Conv2D(80, 1, padding='same', activation='relu6')(x3)
    x3a = layers.Attention(use_scale=True)([x3,x3,x3])
    # x3 = SelfAttention_Block(4)(x3)
    p4 = layers.Add(name='fuse2')([p4, x3, x3a])

    p3 = layers.Conv2DTranspose(40,3,2,padding='same',name='up3')(p4)
    x2 = layers.Concatenate()([x2,x2p])
    x2 = layers.Conv2D(40, 1, padding='same', activation='relu6')(x2)
    x2a = layers.Attention(use_scale=True)([x2,x2,x2])
    # x2 = SelfAttention_Block(4)(x2)
    p3 = layers.Add(name='fuse3')([p3, x2, x2a])

    p2 = layers.Conv2DTranspose(16,3,2,padding='same',name='up2')(p3)
    # p2 = layers.Add(name='fuse4')([p2, layers.Conv2D(16, 1, padding='same', activation='relu6')(x1)])
    # bottom-up augmentation
    n2 = layers.SeparableConv2D(40,3,2,padding='same',name='bottomup1')(p2)
    n2 = layers.Add(name='fuse5')([n2, p3])

    n3 = layers.SeparableConv2D(80,3,2,padding='same',name='bottomup2')(n2)
    n3 = layers.Add(name='fuse6')([n3, p4])
    # n3 = SelfAttention_Block(4, name='attention1')(n3)

    n4 = layers.SeparableConv2D(160,3,2,padding='same',name='bottomup3')(n3)
    n4 = layers.Add(name='fuse7')([n4, p5])
    # n4 = SelfAttention_Block(4, name='attention2')(n4)

    n5 = layers.SeparableConv2D(320,3,2,padding='same',name='bottomup4')(n4)
    out_5 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n5)
    out_5 = layers.Resizing(416,416, name="aux_out4")(out_5)
    n5 = layers.UpSampling2D(8)(n5)
    out_4 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n4)
    out_4 = layers.Resizing(416,416, name="aux_out3")(out_4)
    n4 = layers.UpSampling2D(4)(n4)
    out_3 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n3)
    out_3 = layers.Resizing(416,416, name="aux_out2")(out_3)
    n3 = layers.UpSampling2D(2)(n3)
    out_2 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n2)
    out_2 = layers.Resizing(416,416, name="aux_out1")(out_2)

    out = layers.Concatenate()([n2,n3,n4,n5])
    out = layers.Conv2DTranspose(16,3,2,padding='same',name='out1')(out)
    out = layers.Conv2DTranspose(CLASS_NUM,3,2,padding='same',name='out2')(out)
    out = tf.keras.layers.Softmax(name='softmax_out')(out)

    return tf.keras.Model(inputs=[input], outputs=[out, out_2, out_3, out_4, out_5])

def get_eefm_dual_attention(CLASS_NUM = 3):
    # down sample with CSP
    input = tf.keras.Input((416,416,3),name='input')
    x1 = MNV3_Block(3,16,2,hswish,name='block1')(input)
    x2 = MNV3_Block(5,40,2,name='block2')(x1)
    x2, x2p = tf.split(x2,num_or_size_splits=2, axis=-1)
    x3 = MNV3_Block(3,80,2,hswish,name='block3')(x2)
    x3, x3p = tf.split(x3,num_or_size_splits=2, axis=-1)
    x4 = MNV3_Block(3,160,2,hswish,name='block4')(x3)
    x4, x4p = tf.split(x4,num_or_size_splits=2, axis=-1)
    x5 = MNV3_Block(3,320,2,hswish,name='block5')(x4)
    x5 = DualSelfAttention_Block(identity=True)(x5)
    # up sample
    p5 = layers.Conv2DTranspose(160,3,2,padding='same',name='up5')(x5)
    x4 = layers.Concatenate()([x4,x4p])
    x4a = DualSelfAttention_Block()(x4)
    p5 = layers.Add(name='fuse1')([p5, x4, x4a])

    p4 = layers.Conv2DTranspose(80,3,2,padding='same',name='up4')(p5)
    x3 = layers.Concatenate()([x3,x3p])
    x3a = DualSelfAttention_Block()(x3)
    p4 = layers.Add(name='fuse2')([p4, x3, x3a])

    p3 = layers.Conv2DTranspose(40,3,2,padding='same',name='up3')(p4)
    x2 = layers.Concatenate()([x2,x2p])
    x2a = DualSelfAttention_Block()(x2)
    p3 = layers.Add(name='fuse3')([p3, x2, x2a])

    p2 = layers.Conv2DTranspose(16,3,2,padding='same',name='up2')(p3)

    # bottom-up augmentation
    n2 = layers.SeparableConv2D(40,3,2,padding='same',name='bottomup1')(p2)
    n2 = layers.Add(name='fuse5')([n2, p3])
    n2 = DualSelfAttention_Block(identity=True)(n2)

    n3 = layers.SeparableConv2D(80,3,2,padding='same',name='bottomup2')(n2)
    n3 = layers.Add(name='fuse6')([n3, p4])
    n3 = DualSelfAttention_Block(identity=True)(n3)

    n4 = layers.SeparableConv2D(160,3,2,padding='same',name='bottomup3')(n3)
    n4 = layers.Add(name='fuse7')([n4, p5])
    n4 = DualSelfAttention_Block(identity=True)(n4)

    n5 = layers.SeparableConv2D(320,3,2,padding='same',name='bottomup4')(n4)
    out_5 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n5)
    out_5 = layers.Resizing(416,416, name="aux_out4")(out_5)
    n5 = layers.UpSampling2D(8)(n5)
    out_4 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n4)
    out_4 = layers.Resizing(416,416, name="aux_out3")(out_4)
    n4 = layers.UpSampling2D(4)(n4)
    out_3 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n3)
    out_3 = layers.Resizing(416,416, name="aux_out2")(out_3)
    n3 = layers.UpSampling2D(2)(n3)
    out_2 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n2)
    out_2 = layers.Resizing(416,416, name="aux_out1")(out_2)

    out = layers.Concatenate()([n2,n3,n4,n5])
    out = layers.Conv2DTranspose(16,3,2,padding='same',name='out1')(out)
    out = layers.Conv2DTranspose(CLASS_NUM,3,2,padding='same',name='out2')(out)
    out = tf.keras.layers.Softmax(name='softmax_out')(out)

    return tf.keras.Model(inputs=[input], outputs=[out, out_2, out_3, out_4, out_5])

def get_eefm_cross_attention(CLASS_NUM = 3):
    input = tf.keras.Input((416,416,3),name='input') #3534203934
    # down convolution
    x1 = MNV3_Block(3,16,2,hswish,name='block1')(input)
    x2 = MNV3_Block(5,40,2,name='block2')(x1)
    x2, x2p = tf.split(x2,num_or_size_splits=2, axis=-1)
    x3 = MNV3_Block(3,80,2,hswish,name='block3')(x2)
    x3, x3p = tf.split(x3,num_or_size_splits=2, axis=-1)
    x4 = MNV3_Block(3,160,2,hswish,name='block4')(x3)
    x4, x4p = tf.split(x4,num_or_size_splits=2, axis=-1)
    x5 = MNV3_Block(3,160,2,hswish,name='block5')(x4)
    x5 = DualSelfAttention_Block(identity=True)(x5)

    # up sample
    p5 = layers.Conv2DTranspose(160,3,2,padding='same',name='up5')(x5)
    x4a = layers.Attention(use_scale=True)([p5, x4p, p5])
    x4a = layers.Conv2D(80, 1, padding='same', activation='sigmoid')(x4a)
    x4p = layers.Multiply()([x4a, x4p])
    x4 = layers.Concatenate()([x4, x4p])
    p5 = layers.Add(name='fuse1')([p5, x4])

    p4 = layers.Conv2DTranspose(80,3,2,padding='same',name='up4')(p5)
    x3a = layers.Attention(use_scale=True)([p4, x3p, p4])
    x3a = layers.Conv2D(40, 1, padding='same', activation='sigmoid')(x3a)
    x3p = layers.Multiply()([x3a, x3p])
    x3 = layers.Concatenate()([x3, x3p])
    p4 = layers.Add(name='fuse2')([p4, x3])

    p3 = layers.Conv2DTranspose(40,3,2,padding='same',name='up3')(p4)
    x2a = layers.Attention(use_scale=True)([p3, x2p, p3])
    x2a = layers.Conv2D(20, 1, padding='same', activation='sigmoid')(x2a)
    x2p = layers.Multiply()([x2a, x2p])
    x2 = layers.Concatenate()([x2, x2p])
    p3 = layers.Add(name='fuse3')([p3, x2])

    p2 = layers.Conv2DTranspose(16,3,2,padding='same',name='up2')(p3)
    
    # bottom-up augmentation
    n2 = layers.SeparableConv2D(32,3,2,padding='same',name='bottomup1')(p2)
    n2a = layers.Attention(use_scale=True)([n2, p3, n2])
    n2 = layers.Add(name='fuse5')([n2, n2a, p3])

    n3 = layers.SeparableConv2D(64,3,2,padding='same',name='bottomup2')(n2)
    n3a = layers.Attention(use_scale=True)([n3, p4, n3])
    n3 = layers.Add(name='fuse6')([n3, n3a, p4])

    n4 = layers.SeparableConv2D(128,3,2,padding='same',name='bottomup3')(n3)
    n4a = layers.Attention(use_scale=True)([n4, p5, n4])
    n4 = layers.Add(name='fuse7')([n4, n4a, p5])

    # auxiliary outputs
    n5 = layers.SeparableConv2D(256,3,2,padding='same',name='bottomup4')(n4)
    out_5 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n5)
    out_5 = layers.Resizing(416,416, name="aux_out4")(out_5)
    n5 = layers.UpSampling2D(8)(n5)
    out_4 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n4)
    out_4 = layers.Resizing(416,416, name="aux_out3")(out_4)
    n4 = layers.UpSampling2D(4)(n4)
    out_3 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n3)
    out_3 = layers.Resizing(416,416, name="aux_out2")(out_3)
    n3 = layers.UpSampling2D(2)(n3)
    out_2 = layers.Conv2D(CLASS_NUM, 3, padding='same', activation='relu6')(n2)
    out_2 = layers.Resizing(416,416, name="aux_out1")(out_2)

    # output
    out = layers.Concatenate()([n2,n3,n4,n5])
    out = layers.Conv2DTranspose(16,3,2,padding='same',name='out1')(out)
    out = layers.Conv2DTranspose(CLASS_NUM,3,2,padding='same',name='out2')(out)
    out = tf.keras.layers.Softmax(name='softmax_out')(out)

    return tf.keras.Model(inputs=[input], outputs=[out, out_2, out_3, out_4, out_5])

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

def get_efm_v2(CLASS_NUM = 3):
    input = tf.keras.Input((416,416,3),name='input')
    x1 = MNV3_Block(3,16,2,hswish,name='block1-1')(input)
    x1 = ELAN_Block(24,3,activation=hswish,use_mobile=True,name='block1-2')(x1)
    x2 = MNV3_Block(5,32,2,name='block2')(x1)
    x2, x2p = tf.split(x2,num_or_size_splits=2, axis=-1)
    x3 = MNV3_Block(3,64,2,hswish,name='block3')(x2)
    x3, x3p = tf.split(x3,num_or_size_splits=2, axis=-1)
    x4 = MNV3_Block(3,128,2,hswish,name='block4')(x3)
    x4, x4p = tf.split(x4,num_or_size_splits=2, axis=-1)
    x5 = MNV3_Block(3,256,2,hswish,name='block5-1')(x4)
    x5 = DualSelfAttention_Block(identity=True,name='block5-2')(x5)
    x5 = ELAN_Block(256,3,activation=hswish,use_mobile=True,name='block5-3')(x5)

    # up sample with channel cross attention
    p5 = layers.Conv2DTranspose(128,3,2,padding='same',name='up5')(x5)
    x4 = layers.Concatenate()([x4,x4p])
    p5 = CCA_Block(name='fuse1')([p5,x4])

    p4 = layers.Conv2DTranspose(64,3,2,padding='same',name='up4')(p5)
    x3 = layers.Concatenate()([x3,x3p])
    p4 = CCA_Block(name='fuse2')([p4,x3])

    p3 = layers.Conv2DTranspose(32,3,2,padding='same',name='up3')(p4)
    x2 = layers.Concatenate()([x2,x2p])
    p3 = CCA_Block(name='fuse3')([p3,x2])
    
    # bottom-up augmentation
    n1 = layers.Conv2DTranspose(24,3,2,padding='same',name='up2')(p3)
    x1 = Conv_Block(3,24,1,activation='relu6')(x1)
    n1 = layers.Add(name='fuse4')([n1, x1])

    n2 = layers.SeparableConv2D(32,3,2,padding='same',name='bottomup1')(n1)
    x2 = Conv_Block(3,32,1,activation='relu6')(x2)
    n2 = CCA_Block(name='fuse5')([p3, n2])
    n2 = layers.Add()([x2, n2])

    n3 = layers.SeparableConv2D(64,3,2,padding='same',name='bottomup2')(n2)
    x3 = Conv_Block(3,64,1,activation='relu6')(x3)
    n3 = CCA_Block(name='fuse6')([p4, n3])
    n3 = layers.Add()([x3, n3])

    n4 = layers.SeparableConv2D(128,3,2,padding='same',name='bottomup3')(n3)
    x4 = Conv_Block(3,128,1,activation='relu6')(x4)
    n4 = CCA_Block(name='fuse7')([p5, n4])
    n4 = layers.Add()([x4, n4])

    n5 = layers.SeparableConv2D(256,3,2,padding='same',name='bottomup4')(n4)
    x5 = Conv_Block(3,256,1,activation='relu6')(x5)
    n5 = layers.Add()([x5, n5])
    n5 = ELAN_Block(256,3,activation=hswish,use_mobile=True,name='bottomup4-2')(n5)

    # auxiliary outputs
    out_5 = Conv_Block(3,CLASS_NUM,1,activation='relu6')(n5)
    out_5 = layers.Resizing(416,416, name="aux_out4")(out_5)
    out_4 = Conv_Block(3,CLASS_NUM,1,activation='relu6')(n4)
    out_4 = layers.Resizing(416,416, name="aux_out3")(out_4)
    out_3 = Conv_Block(3,CLASS_NUM,1,activation='relu6')(n3)
    out_3 = layers.Resizing(416,416, name="aux_out2")(out_3)
    out_2 = Conv_Block(3,CLASS_NUM,1,activation='relu6')(n2)    
    out_2 = layers.Resizing(416,416, name="aux_out1")(out_2)

    # outputs
    n5 = layers.UpSampling2D(8)(n5)
    n4 = layers.UpSampling2D(4)(n4)
    n3 = layers.UpSampling2D(2)(n3)
    out = layers.Concatenate()([n2,n3,n4,n5])
    out = layers.Conv2D(128, 1, padding='same', activation='relu6')(out)
    # out = layers.Conv2D(64, 1, padding='same', activation='relu6')(out)
    # out = layers.Conv2D(32, 1, padding='same', activation='relu6')(out)
    out = layers.Conv2DTranspose(16,3,2,padding='same',name='out1')(out)
    out = layers.Conv2DTranspose(CLASS_NUM,3,2,padding='same',name='out2')(out)
    out = tf.keras.layers.Softmax(name='softmax_out')(out)

    return tf.keras.Model(inputs=[input], outputs=[out, out_2, out_3, out_4, out_5])

def get_unet_transformer(CLASS_NUM = 3, HEADS = 3):
    input = tf.keras.Input((416,416,3))
    x1 = UnetDown_Block(64)(input)
    x2 = UnetDown_Block(128)(layers.MaxPool2D()(x1))
    x3 = UnetDown_Block(256)(layers.MaxPool2D()(x2))
    x4 = layers.MaxPool2D()(x3)
    x4 = MHSA_Block(HEADS)(x4)
    up1 = MHCA_Block(HEADS)([x3,x4])
    up1 = Conv3_Block(256)(up1)
    up2 = MHCA_Block(HEADS)([x2,up1])
    up2 = Conv3_Block(128)(up2)
    up3 = MHCA_Block(HEADS)([x1,up2])
    up3 = Conv3_Block(64)(up3)
    out = Conv3_Block(CLASS_NUM)(up3)
    return tf.keras.Model(inputs=[input], outputs=[out])

def get_munet_transformer(CLASS_NUM = 3, HEADS = 3):
    input = tf.keras.Input((416,416,3))
    x1 = MNV3_Block(3,64,1)(input)
    x2 = MNV3_Block(3,128,1)(layers.MaxPool2D()(x1))
    x3 = MNV3_Block(3,256,1)(layers.MaxPool2D()(x2))
    x4 = layers.MaxPool2D()(x3)
    x4 = MHSA_Block(HEADS)(x4)
    up1 = MHCA_Block(HEADS)([x3,x4])
    up1 = Conv_Block(3,256,1)(up1)
    up2 = MHCA_Block(HEADS)([x2,up1])
    up2 = Conv_Block(3,128,1)(up2)
    up3 = MHCA_Block(HEADS)([x1,up2])
    up3 = Conv_Block(3,64,1)(up3)
    out = Conv_Block(3,CLASS_NUM,1)(up3)
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
    
# IoU metric for sparse category, IoU = TP/(TP+FP+FN)
class SparseMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
               num_classes=None,
               name='sparse_mean_iou',
               dtype=None):
        super(SparseMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

class SparseCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    # def __init__(self,
    #             n_classes,
    #             **kwargs):
    #     self.n_classes = n_classes
    #     super(SparseCategoricalCrossentropy, self).__init__(**kwargs)

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'n_classes': self.n_classes
    #     })
    #     return config

    def call(self, y_true, y_pred):
        n_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(tf.cast(tf.squeeze(y_true,axis=-1),tf.int32), n_classes, axis=-1)
        return super(SparseCategoricalCrossentropy, self).call(y_true, y_pred)

def contour(x):
    '''
    Differenciable aproximation of contour extraction
    
    '''   
    min_pool_x = tf.nn.max_pool2d(x*-1, (3, 3), 1, 'SAME')*-1
    max_min_pool_x = tf.nn.max_pool2d(min_pool_x, (3, 3), 1, 'SAME')
    contour = tf.nn.relu(max_min_pool_x - min_pool_x)
    return contour

class BoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, classes, sparse=True, from_logits=False, name='boundary_loss', **kwargs):
        super(BoundaryLoss, self).__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.sparse = sparse
        self.classes = classes

    def get_config(self):
        config = {
            'from_logits': self.from_logits,
            'sparse': self.sparse,
            'classes': self.classes,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, y_true, y_pred):
        shape = tf.cast(tf.shape(y_true), tf.float32)
        w = shape[1]
        h = shape[2]
        if (self.from_logits):
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            y_true = tf.nn.softmax(y_true, axis=-1)
        if (self.sparse):
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
        else:
            y_pred = tf.cast(tf.squeeze(y_pred, axis=-1), tf.int32)
            y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_pred = tf.one_hot(y_pred, self.classes)
        y_true = tf.one_hot(y_true, self.classes)
        pred = tf.reduce_sum(contour(y_pred),axis=[1,2])
        true = tf.reduce_sum(contour(y_true),axis=[1,2])
        loss = tf.sqrt((pred - true) ** 2)/(w*h)
        return tf.reduce_mean(loss)
    
class DiceBoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, classes, smooth=1, sparse=True, from_logits=False, name='dice_boundary_loss', **kwargs):
        super(DiceBoundaryLoss, self).__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.sparse = sparse
        self.classes = classes
        self.smooth = float(smooth)

    def get_config(self):
        config = {
            'from_logits': self.from_logits,
            'sparse': self.sparse,
            'classes': self.classes,
            'smooth': self.smooth,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, y_true, y_pred):
        if (self.from_logits):
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            y_true = tf.nn.softmax(y_true, axis=-1)
        if (self.sparse):
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
        else:
            y_pred = tf.cast(tf.squeeze(y_pred, axis=-1), tf.int32)
            y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_pred = tf.one_hot(y_pred, self.classes, dtype=tf.float32)
        y_true = tf.one_hot(y_true, self.classes, dtype=tf.float32)
        pred = contour(y_pred)
        true = contour(y_true)
        intersect = tf.reduce_sum(pred*true,axis=[1,2])
        loss = 1.0-(2.0*intersect+self.smooth)/(tf.reduce_sum(pred,axis=[1,2])+tf.reduce_sum(true,axis=[1,2])+self.smooth)
        return tf.reduce_mean(loss)