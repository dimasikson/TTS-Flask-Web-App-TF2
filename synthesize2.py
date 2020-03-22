
import os
from hyperparams import Hyperparams as hp
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from networks2 import Text2Mel, SSRN
from modules2 import C_Alignment_Loss, C_Reduce_Mean, C_Reduce_Mean_Logits, C_Empty_Loss
import time

def synthesize(text_inputs, ttm_model_num=-1, ssrn_model_num=-1, mono_attention=True):
    L = load_data(mode="synth", text_inputs=text_inputs)

    model_folder_name = "finished_models - tts2 current best"

    TTM_fnames = []
    SSRN_fnames = []
    for filename in os.listdir(os.path.join(hp.data, model_folder_name)):
        if filename.startswith("model_TTM"):
            TTM_fnames.append(filename)
        elif filename.startswith("model_SSRN"):
            SSRN_fnames.append(filename)

    TTM_fname = TTM_fnames[ttm_model_num]
    SSRN_fname = SSRN_fnames[ssrn_model_num]

    Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
    Z = np.zeros((len(L), hp.max_T*4, 1+hp.n_fft//2), np.float32)
    L = tf.convert_to_tensor(L)
    align = tf.zeros((len(L), hp.max_N, hp.max_T), tf.float32)
    max_att_loss = tf.zeros((len(L), hp.max_T), tf.float32)
    prev_max_attentions = np.zeros((len(L),), np.int32)

    model_TTM = Text2Mel(training=False, mono_attention=mono_attention)
    model_SSRN = SSRN(training=False)

    model_TTM.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.lr, clipvalue=1.),
                      loss=[C_Reduce_Mean(),
                            C_Reduce_Mean_Logits(),
                            C_Alignment_Loss(),
                            C_Empty_Loss()],
                      loss_weights=[float(hp.ttm_rmean),
                                    float(hp.ttm_rmean_logits),
                                    float(hp.ttm_rsum_align),
                                    float(0)])


    model_SSRN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.lr, clipvalue=1.),
                       loss=[C_Reduce_Mean(),
                             C_Reduce_Mean_Logits()],
                       loss_weights=[float(hp.ssrn_rmean),
                                     float(hp.ssrn_rmean_logits)])

    model_TTM.train_on_batch(x=[L, Y, prev_max_attentions],
                             y=[Y, Y, align, max_att_loss])

    model_SSRN.train_on_batch(x=[Y],
                              y=[Z,Z])

    model_TTM.load_weights(os.path.join(hp.data, model_folder_name, TTM_fname))
    model_SSRN.load_weights(os.path.join(hp.data, model_folder_name, SSRN_fname))

    for j in range(hp.max_T):
        stft_num = str(j)

        S = tf.concat((tf.zeros_like(Y[:, :1, :]), Y[:, :-1, :]), 1)
        _Y, _Y_logits, _alignments, _max_attentions = model_TTM.predict([L, S, prev_max_attentions])
        Y[:, j, :] = _Y[:, j, :]
        prev_max_attentions = _max_attentions[:, j]

        if j%50==0:
            print(j)

    timestamp = time.strftime("%Y%m%d%H%M%S")

    Z, Z_logits = model_SSRN.predict(Y)

    # Generate wav files
    wav = spectrogram2wav(Z[0])
    #write(os.path.join(hp.data, "outputs", "Sample_wav_"+timestamp+".wav"),hp.sr, wav)

    return wav
