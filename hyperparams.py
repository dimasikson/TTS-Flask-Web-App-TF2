# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

import math

class Hyperparams:
    '''Hyper parameters'''
    # pipeline
    prepro = False  # if True, run `python prepro.py` first before running `python train.py`.

    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20

    # Model
    r = 4  # Reduction factor. Do not change this.
    dropout_rate = 0.05
    e = 128  # == embedding
    d = 256  # == hidden units of Text2Mel
    c = 512  # == hidden units of SSRN
    attention_win_size = 3

    # data
    data = r"C:\Users\dimaz\Documents\dc_tts-master\dc_tts-tfv2\LJSpeech-1.1\LJSpeech-1.1"
    test_data = 'harvard_sentences.txt'
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
    max_N = 180  # Maximum number of characters.
    max_T = 210  # Maximum number of mel frames.

    # training scheme
    lr = 0.001  # Initial learning rate.
    decay = 0.0001
    beta1 = 0.5
    beta2 = 0.9
    epsilon = 1e-06
    l2 = 0.000001
    logdir = "logdir/LJ01"
    sampledir = 'samples'
    B = 4  # batch size
    warmup_epochs = 3

    num_iterations = 2000000
    epochs = 100

    # loss weights
    # TTM
    ttm_rmean = 1
    ttm_rmean_logits = 1
    ttm_rsum_align = 2

    # SSRN
    ssrn_rmean = 1
    ssrn_rmean_logits = 1
