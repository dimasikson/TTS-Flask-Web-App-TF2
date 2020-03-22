# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
from modules2 import C_Conv1d, C_Conv1d_Transpose, C_HC, C_Attention, C_Split
import tensorflow as tf

def Text2Mel(training=True, mono_attention=False):
    #TEXTENC
    i = 1
    L_input = tf.keras.layers.Input(shape=(None,), dtype=tf.float32, name="TextEnc"+str(i)); i += 1

    L = tf.keras.layers.Embedding(input_dim=len(hp.vocab),
                                  output_dim=hp.e,
                                  embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                                  mask_zero=True,
                                  name="TextEnc"+str(i))(L_input); i += 1

    L = C_Conv1d(filters=2*hp.d,
                 kernel_size=1,
                 dilation_rate=1,
                 dropout_rate=hp.dropout_rate,
                 activation_fn=tf.nn.relu,
                 training=training,
                 layer_name="TextEnc"+str(i))(L); i += 1


    L = C_Conv1d(filters=L.get_shape().as_list()[-1],
                 kernel_size=1,
                 dilation_rate=1,
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="TextEnc"+str(i))(L); i += 1

    for _ in range(2):
        for j in range(4):
            L = C_HC(filters=L.get_shape().as_list()[-1],
                     kernel_size=3,
                     dilation_rate=3 ** j,
                     dropout_rate=hp.dropout_rate,
                     activation_fn=None,
                     training=training,
                     layer_name="TextEnc"+str(i))(L); i += 1

    for _ in range(2):
        L = C_HC(filters=L.get_shape().as_list()[-1],
                 kernel_size=3,
                 dilation_rate=1,
                 dropout_rate=hp.dropout_rate,
                 activation_fn=None,
                 training=training,
                 layer_name="TextEnc"+str(i))(L); i += 1

    for _ in range(2):
        L = C_HC(filters=L.get_shape().as_list()[-1],
                 kernel_size=1,
                 dilation_rate=1,
                 dropout_rate=hp.dropout_rate,
                 activation_fn=None,
                 training=training,
                 layer_name="TextEnc"+str(i))(L); i += 1

    K, V = C_Split(axis=-1)(L)

    #AUDIOENC
    S_input = tf.keras.layers.Input(shape=(None,80), dtype=tf.float32, name="AudioEnc"+str(i)); i += 1

    S = C_Conv1d(filters=hp.d,
                 kernel_size=1,
                 dilation_rate=1,
                 padding="CAUSAL",
                 dropout_rate=hp.dropout_rate,
                 activation_fn=tf.nn.relu,
                 training=training,
                 layer_name="AudioEnc"+str(i))(S_input); i += 1

    S = C_Conv1d(filters=S.get_shape().as_list()[-1],
                 kernel_size=1,
                 dilation_rate=1,
                 padding="CAUSAL",
                 dropout_rate=hp.dropout_rate,
                 activation_fn=tf.nn.relu,
                 training=training,
                 layer_name="AudioEnc"+str(i))(S); i += 1
    S = C_Conv1d(filters=S.get_shape().as_list()[-1],
                 kernel_size=1,
                 dilation_rate=1,
                 padding="CAUSAL",
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="AudioEnc"+str(i))(S); i += 1

    for _ in range(2):
        for j in range(4):
            S = C_HC(filters=S.get_shape().as_list()[-1],
                     kernel_size=3,
                     dilation_rate=3 ** j,
                     padding="CAUSAL",
                     dropout_rate=hp.dropout_rate,
                     training=training,
                     layer_name="AudioEnc"+str(i))(S); i += 1

    S = C_HC(filters=S.get_shape().as_list()[-1],
             kernel_size=3,
             dilation_rate=3,
             padding="CAUSAL",
             dropout_rate=hp.dropout_rate,
             training=training,
             layer_name="AudioEnc"+str(i))(S); i += 1
    Q = C_HC(filters=S.get_shape().as_list()[-1],
             kernel_size=3,
             dilation_rate=3,
             padding="CAUSAL",
             dropout_rate=hp.dropout_rate,
             training=training,
             layer_name="AudioEnc"+str(i))(S); i += 1

    #ATTENTION
    prev_max_attentions = tf.keras.layers.Input(shape=(), dtype=tf.float32, name="Attention"+str(i)); i += 1

    R, alignments, max_attentions = C_Attention(training=training, mono_attention=mono_attention)(Q, K, V, prev_max_attentions); i += 1

    #AUDIODEC

    R = C_Conv1d(filters=hp.d,
                 kernel_size=1,
                 dilation_rate=1,
                 padding="CAUSAL",
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="AudioDec"+str(i))(R); i += 1
    for j in range(4):
        R = C_HC(filters=R.get_shape().as_list()[-1],
                 kernel_size=3,
                 dilation_rate=3 ** j,
                 padding="CAUSAL",
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="AudioDec"+str(i))(R); i += 1

    for _ in range(2):
        R = C_HC(filters=R.get_shape().as_list()[-1],
                 kernel_size=3,
                 dilation_rate=1,
                 padding="CAUSAL",
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="AudioDec"+str(i))(R); i += 1

    for _ in range(3):
        R = C_Conv1d(filters=R.get_shape().as_list()[-1],
                     kernel_size=1,
                     dilation_rate=1,
                     padding="CAUSAL",
                     dropout_rate=hp.dropout_rate,
                     activation_fn=tf.nn.relu,
                     training=training,
                     layer_name="AudioDec"+str(i))(R); i += 1

    # mel_hats
    logits = C_Conv1d(filters=hp.n_mels,
                      kernel_size=1,
                      dilation_rate=1,
                      padding="CAUSAL",
                      dropout_rate=hp.dropout_rate,
                      training=training,
                      layer_name="AudioDec"+str(i))(R); i += 1

    Y = tf.nn.sigmoid(logits) # mel_hats

    model_TTM = tf.keras.Model(inputs=[L_input, S_input, prev_max_attentions],
                               outputs=[Y,
                                        logits,
                                        alignments,
                                        max_attentions])

    return model_TTM

def SSRN(training=True):
    #SSRN
    i = 1
    Y_input = tf.keras.layers.Input(shape=(None,80), dtype=tf.float32, name="layer"+str(i)); i += 1

    Y = C_Conv1d(filters=hp.c,
                 kernel_size=1,
                 dilation_rate=1,
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="layer" + str(i))(Y_input); i += 1

    for j in range(2):
        Y = C_HC(filters=Y.get_shape().as_list()[-1],
                 kernel_size=3,
                 dilation_rate=3**j,
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="layer"+str(i))(Y); i += 1

    for _ in range(2):
        # -> (B, T/2, c) -> (B, T, c)
        Y = C_Conv1d_Transpose(filters=Y.get_shape().as_list()[-1],
                               dropout_rate=hp.dropout_rate,
                               training=training)(Y); i += 1

        for j in range(2):
            Y = C_HC(filters=Y.get_shape().as_list()[-1],
                     kernel_size=3,
                     dilation_rate=3 ** j,
                     dropout_rate=hp.dropout_rate,
                     training=training,
                     layer_name="layer" + str(i))(Y); i += 1

    # -> (B, T, 2*c)
    Y = C_Conv1d(filters=2*hp.c,
                 kernel_size=1,
                 dilation_rate=1,
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="layer" + str(i))(Y); i += 1

    for _ in range(2):
    # -> (B, T, 1+n_fft/2)
        Y = C_HC(filters=Y.get_shape().as_list()[-1],
                 kernel_size=3,
                 dilation_rate=1,
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="layer" + str(i))(Y); i += 1

    Y = C_Conv1d(filters=1+hp.n_fft//2,
                 kernel_size=1,
                 dilation_rate=1,
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="layer" + str(i))(Y); i += 1

    for _ in range(2):
        Y = C_Conv1d(filters=Y.get_shape().as_list()[-1],
                     kernel_size=1,
                     dilation_rate=1,
                     dropout_rate=hp.dropout_rate,
                     activation_fn=tf.nn.relu,
                     training=training,
                     layer_name="layer" + str(i))(Y); i += 1

    logits = C_Conv1d(filters=Y.get_shape().as_list()[-1],
                 kernel_size=1,
                 dilation_rate=1,
                 dropout_rate=hp.dropout_rate,
                 training=training,
                 layer_name="layer" + str(i))(Y); i += 1

    Z = tf.nn.sigmoid(logits)

    model_SSRN = tf.keras.Model(inputs=[Y_input],
                                outputs=[Z, logits])

    return model_SSRN
