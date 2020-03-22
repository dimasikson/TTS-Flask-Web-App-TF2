from __future__ import print_function

import os
import time

from hyperparams import Hyperparams as hp
import numpy as np
import pandas as pd
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm
import datetime

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


def init(): 
	g = Graph(mode="synthesize")
	return g

def synthesize(text_input):
    # Load data
    L = load_data(mode="synthesize",text_input=text_input)

    g = Graph(mode="synthesize")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(os.path.join(hp.logdir) + "-1" + " - Copy"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(os.path.join(hp.logdir) + "-2" + " - Copy"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)

        print(datetime.datetime.now())

        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _Y_logits, _max_attentions, _alignments = sess.run([g.global_step, g.Y, g.Y_logits, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]
            if j % 1 == 0:
                stft_num = str(j)

        timestamp = time.strftime("%Y%m%d%H%M%S")

        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        for i, mag in enumerate(Z):
            print("Working on file", i+1)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            wav = spectrogram2wav(Z[0])

            filename = "Sample_wav_"+timestamp+".wav"
            write(os.path.join('static',filename),hp.sr, wav)

        return filename
