# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts

'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
import tqdm
import time

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train", text_inputs=""):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode=="train":
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'metadata.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()

            for line in lines:
                fname, _, text = line.strip().split("|")
                if fname[0] == '"':
                    fname = fname[1:]

                fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                fpaths.append(fpath)

                text = text_normalize(text) # E: EOS
                if text[-1] in ["'", ".", "?", " "]:
                    text = text[:-1] + " E"
                else:
                    text = text + " E"

                text = [char2idx[char] for char in text]
                text_len = min(len(text),hp.max_N)
                text_lengths.append(text_len)
                texts.append(np.array(text, np.int32))

            maxlen, minlen = max(text_lengths), min(text_lengths)

            # Calc total batch count
            num_batch = len(fpaths) // hp.B

            text_lengths = np.asarray(text_lengths)
            texts = np.asarray(texts)

            return fpaths, text_lengths, texts, maxlen, minlen

    else: # synthesize on unseen test text.
        # Parse
        lines = [text_inputs]
        sents = [text_normalize(line).strip() for line in lines] # text normalization, E: EOS

        for i in range(len(sents)):
            if sents[i][-1] in ["'",".","?"," "]:
                sents[i] = sents[i][:-1] + " E"
            else:
                sents[i] = sents[i] + " E"

        texts = np.zeros((len(sents), hp.max_N), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts

def get_batch(fpaths):
    """Loads training data and put them in queues"""
    fname, mel, mag = [], [], []
    max_mel = 0
    max_mag = 0

    for fpath in fpaths:
        fname1, mel1, mag1 = load_spectrograms(fpath)  # (None, n_mels)
        fname.append(fname1)
        mel.append(mel1)
        mag.append(mag1)
        if mel1.shape[0] > max_mel:
            max_mel = mel1.shape[0]
        if mag1.shape[0] > max_mag:
            max_mag = mag1.shape[0]

    # Batching
    for n in range(len(fname)):
        mel[n] = np.pad(mel[n], [[0, max_mel-mel[n].shape[0]], [0, 0]], mode="constant")
        mag[n] = np.pad(mag[n], [[0, max_mag-mag[n].shape[0]], [0, 0]], mode="constant")

    fnames = tf.convert_to_tensor(fname)
    mels = tf.convert_to_tensor(mel)
    mags = tf.convert_to_tensor(mag)

    return mels, mags, fnames, max_mel, max_mag

