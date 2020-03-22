
from __future__ import print_function, division

from hyperparams import Hyperparams as hp
import tensorflow as tf
from utils import *


class C_Conv1d(tf.keras.layers.Layer):
    def __init__(self,
                 filters=0,
                 kernel_size=1,
                 dilation_rate=1,
                 padding="SAME",
                 dropout_rate=0,
                 use_bias=True,
                 activation_fn=None,
                 training=True,
                 reuse=None,
                 layer_name=None):
        super(C_Conv1d, self).__init__()

        self.filters = filters
        self.kernel_size=kernel_size
        self.rate=dilation_rate
        self.padding=padding
        self.dropout_rate=dropout_rate
        self.use_bias=use_bias
        self.activation_fn=activation_fn
        self.training=training
        self.reuse=reuse

        self.layer_name = "anon_layer"

        self.pad_len = (self.kernel_size - 1) * self.rate  # padding size

        if layer_name!=None:
            self.layer_name = layer_name

        if self.padding.lower() == "causal":
            self.padding = "valid"

        self.conv1d = tf.keras.layers.Conv1D(filters=self.filters,
                                             kernel_size=self.kernel_size,
                                             dilation_rate=self.rate,
                                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.),
                                             #kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.,max_value=1.,rate=1.),
                                             #bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.,max_value=1.,rate=1.),
                                             #kernel_regularizer=tf.keras.regularizers.l2(hp.l2),
                                             #bias_regularizer=tf.keras.regularizers.l2(hp.l2),
                                             padding=self.padding,
                                             use_bias=self.use_bias,
                                             name=self.layer_name)
        self.layernorm = tf.keras.layers.LayerNormalization(axis=-1
                                                            #, gamma_regularizer=tf.keras.regularizers.l2(hp.l2)
                                                            #, beta_regularizer=tf.keras.regularizers.l2(hp.l2)
                                                            #, gamma_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1., max_value=1., rate=1.)
                                                            #, beta_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1., max_value=1., rate=1.)
                                                            )
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        inputs1 = inputs

        if self.padding.lower() == "valid":
            inputs1 = tf.pad(inputs1, [[0, 0], [self.pad_len, 0], [0, 0]])

        inputs1 = self.conv1d(inputs1)

        inputs1 = self.layernorm(inputs1)
        inputs1 = self.activation_fn(inputs1) if self.activation_fn is not None else inputs1
        inputs1 = self.dropout(inputs1,training=self.training)

        return inputs1

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'rate': self.rate,
            'padding': self.padding,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'reuse': self.reuse,
            'activation_fn': self.activation_fn,
            'training': self.training,
            'layer_name': self.layer_name,
            'pad_len': self.pad_len,
            'conv1d': self.conv1d,
            'layernorm': self.layernorm,
            'dropout': self.dropout
        })
        return config


class C_Conv1d_Transpose(tf.keras.layers.Layer):
    def __init__(self,
                 filters=0,
                 kernel_size=3,
                 strides=2,
                 padding="SAME",
                 dropout_rate=0,
                 use_bias=True,
                 activation_fn=None,
                 training=True,
                 reuse=None,
                 layer_name=None):
        super(C_Conv1d_Transpose, self).__init__()

        self.filters=filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.dropout_rate=dropout_rate
        self.use_bias=use_bias
        self.activation_fn=activation_fn
        self.training=training
        self.reuse=reuse

        self.layer_name = "anon_layer"

        if layer_name != None:
            self.layer_name = layer_name

        self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(filters=self.filters,
                                                                kernel_size=(1,self.kernel_size),
                                                                strides=(1,self.strides),
                                                                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.),
                                                                #kernel_regularizer=tf.keras.regularizers.l2(hp.l2),
                                                                #bias_regularizer=tf.keras.regularizers.l2(hp.l2),
                                                                #kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1., max_value=1., rate=1.),
                                                                #bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1., max_value=1., rate=1.),
                                                                padding=self.padding,
                                                                use_bias=self.use_bias,
                                                                name=self.layer_name)
        self.layernorm = tf.keras.layers.LayerNormalization(axis=-1
                                                            #, gamma_regularizer=tf.keras.regularizers.l2(hp.l2)
                                                            #, beta_regularizer=tf.keras.regularizers.l2(hp.l2)
                                                            #, gamma_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1., max_value=1., rate=1.)
                                                            #, beta_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1., max_value=1., rate=1.)
                                                            )
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        inputs1 = inputs

        inputs1 = tf.expand_dims(inputs1,1)
        inputs1 = self.conv2d_transpose(inputs1)
        inputs1 = tf.squeeze(inputs1,1)

        inputs1 = self.layernorm(inputs1)
        inputs1 = self.activation_fn(inputs1) if self.activation_fn is not None else inputs1
        inputs1 = self.dropout(inputs1,training=self.training)

        return inputs1

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'reuse': self.reuse,
            'activation_fn': self.activation_fn,
            'training': self.training,
            'layer_name': self.layer_name,
            'conv2d_transpose': self.conv2d_transpose,
            'layernorm': self.layernorm,
            'dropout': self.dropout
        })
        return config


class C_HC(tf.keras.layers.Layer):
    def __init__(self,
                 filters=0,
                 kernel_size=1,
                 dilation_rate=1,
                 padding="SAME",
                 dropout_rate=0,
                 use_bias=True,
                 activation_fn=None,
                 training=True,
                 reuse=None,
                 layer_name=None):
        super(C_HC, self).__init__()

        self.filters = filters*2
        self.kernel_size=kernel_size
        self.rate=dilation_rate
        self.padding=padding
        self.dropout_rate=dropout_rate
        self.use_bias=use_bias
        self.activation_fn=activation_fn
        self.training=training
        self.reuse=reuse

        self.layer_name = "anon_layer"

        self.pad_len = (self.kernel_size - 1) * self.rate  # padding size

        if layer_name != None:
            self.layer_name = layer_name

        if self.padding.lower() == "causal":
            self.padding = "valid"

        self.conv1d = tf.keras.layers.Conv1D(filters=self.filters,
                                             kernel_size=self.kernel_size,
                                             dilation_rate=self.rate,
                                             kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.),
                                             #kernel_regularizer=tf.keras.regularizers.l2(hp.l2),
                                             #bias_regularizer=tf.keras.regularizers.l2(hp.l2),
                                             #kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.,max_value=1.,rate=1.),
                                             #bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.,max_value=1.,rate=1.),
                                             padding=self.padding,
                                             use_bias=self.use_bias,
                                             name=self.layer_name)
        self.layernorm1 = tf.keras.layers.LayerNormalization(axis=-1
                                                             #, gamma_regularizer=tf.keras.regularizers.l2(hp.l2)
                                                             #, beta_regularizer=tf.keras.regularizers.l2(hp.l2)
                                                             #, gamma_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.,max_value=1.,rate=1.)
                                                             #, beta_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.,max_value=1.,rate=1.)
                                                             )
        self.layernorm2 = tf.keras.layers.LayerNormalization(axis=-1
                                                             #, gamma_regularizer=tf.keras.regularizers.l2(hp.l2)
                                                             #, beta_regularizer=tf.keras.regularizers.l2(hp.l2)
                                                             #, gamma_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.,max_value=1.,rate=1.)
                                                             #, beta_constraint=tf.keras.constraints.MinMaxNorm(min_value=-1.,max_value=1.,rate=1.)
                                                             )
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        _inputs = inputs
        inputs1 = inputs

        if self.padding.lower() == "valid":
            inputs1 = tf.pad(inputs1, [[0, 0], [self.pad_len, 0], [0, 0]])

        inputs1 = self.conv1d(inputs1)

        H1, H2 = tf.split(inputs1, 2, axis=-1)
        H1 = self.layernorm1(H1)
        H2 = self.layernorm2(H2)
        H1 = tf.nn.sigmoid(H1)
        H2 = self.activation_fn(H2) if self.activation_fn is not None else H2
        inputs1 = H1 * H2 + (1. - H1) * _inputs

        inputs1 = self.dropout(inputs1,training=self.training)

        return inputs1

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'rate': self.rate,
            'padding': self.padding,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'reuse': self.reuse,
            'activation_fn': self.activation_fn,
            'training': self.training,
            'layer_name': self.layer_name,
            'pad_len': self.pad_len,
            'conv1d': self.conv1d,
            'layernorm1': self.layernorm2,
            'layernorm2': self.layernorm2,
            'dropout': self.dropout
        })
        return config


class C_Attention(tf.keras.layers.Layer):
    def __init__(self, training=True, mono_attention=False):
        super(C_Attention, self).__init__()
        self.training=training
        self.mono_attention=mono_attention

    def call(self, Q, K, V, prev_max_attentions=None):
        Q, K, V = Q, K, V

        A = tf.matmul(Q, K, transpose_b=True) * tf.math.rsqrt(tf.dtypes.cast(hp.d, tf.float32))
        if self.mono_attention==True:
            key_masks = tf.sequence_mask(prev_max_attentions, hp.max_N)
            reverse_masks = tf.sequence_mask(hp.max_N - hp.attention_win_size - prev_max_attentions, hp.max_N)[:, ::-1]
            masks = tf.logical_or(key_masks, reverse_masks)
            masks = tf.tile(tf.expand_dims(masks, 1), [1, hp.max_T, 1])
            paddings = tf.ones_like(A) * (-2 ** 32 + 1)  # (B, T/r, N)
            A = tf.where(tf.equal(masks, False), A, paddings)
        A = tf.nn.softmax(A)  # (B, T/r, N)
        max_attentions = tf.dtypes.cast(tf.math.argmax(A, -1, output_type=tf.dtypes.int64),tf.float32)  # (B, T/r)
        R = tf.matmul(A, V)
        R = tf.concat((R, Q), -1)

        alignments = tf.transpose(A, [0, 2, 1])  # (B, N, T/r)

        return R, alignments, max_attentions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'training': self.training
        })
        return config


class C_Split(tf.keras.layers.Layer):
    def __init__(self, axis):
        super(C_Split, self).__init__()
        self.axis = axis

    def call(self, inputs):
        K, V = tf.split(inputs, 2, self.axis)
        return K, V

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis
        })
        return config


class C_Alignment_Loss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO):
        super(C_Alignment_Loss, self).__init__(reduction=reduction)

    def call(self, y_true, y_pred):
        gts = tf.convert_to_tensor(guided_attention())
        A = tf.pad(y_pred, [(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT", constant_values=-1.)[:, :hp.max_N, :hp.max_T]
        attention_masks = tf.dtypes.cast(tf.not_equal(A, -1),tf.float32)
        loss_att = tf.reduce_sum(tf.abs(A * gts) * attention_masks)
        mask_sum = tf.reduce_sum(attention_masks)
        loss_att /= mask_sum

        return loss_att

class C_Reduce_Mean(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO):
        super(C_Reduce_Mean, self).__init__(reduction=reduction)

    def call(self, y_true, y_pred):
        loss_mels = tf.math.reduce_mean(tf.abs(y_pred - y_true))

        return loss_mels

class C_Reduce_Mean_Logits(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO):
        super(C_Reduce_Mean_Logits, self).__init__(reduction=reduction)

    def call(self, y_true, y_pred):
        loss_log = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))

        return loss_log

class C_Empty_Loss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO):
        super(C_Empty_Loss, self).__init__(reduction=reduction)

    def call(self, y_true, y_pred):
        return float(0)
