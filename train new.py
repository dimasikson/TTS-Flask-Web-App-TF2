
from __future__ import print_function

import os
import time
from tensorflow.keras import backend as Ker
from data_load import load_data, get_batch, load_vocab
from hyperparams import Hyperparams as hp
from networks2 import Text2Mel, SSRN
from modules2 import C_Alignment_Loss, C_Reduce_Mean, C_Reduce_Mean_Logits, C_Empty_Loss
import tensorflow as tf
from utils import *
import sys
import pandas as pd
import numpy as np
import datetime


def train_TTM(ttm_rmean=hp.ttm_rmean,
              ttm_rmean_logits=hp.ttm_rmean_logits,
              ttm_rsum_align=hp.ttm_rsum_align,
              continue_training=False,
               continue_ttm=-1):
    fpaths, text_lengths, texts, maxlen, minlen = load_data()  # list

    print("Texts loaded")

    np_ar = np.transpose(np.array([fpaths, text_lengths, texts]))

    prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
    align = tf.zeros((hp.B, hp.max_N, hp.max_T), tf.float32)
    max_att_loss = tf.zeros((hp.B, hp.max_T), tf.float32)

    # model build
    model_TTM = Text2Mel()

    print("TTM Model loaded")

    model_TTM.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.lr,
                                                                 clipvalue=1.,
                                                                 beta_1=hp.beta1,
                                                                 beta_2=hp.beta2,
                                                                 epsilon=hp.epsilon),
                      loss=[C_Reduce_Mean(),
                            C_Reduce_Mean_Logits(),
                            C_Alignment_Loss(),
                            C_Empty_Loss()],
                      loss_weights=[float(ttm_rmean),
                                    float(ttm_rmean_logits),
                                    float(ttm_rsum_align),
                                    float(0)])

    gv = 0
    start_epoch = 1

    if continue_training==True:
        TTM_fnames = []
        for filename in os.listdir(os.path.join(hp.data, "finished_models")):
            if filename.startswith("model_TTM"):
                TTM_fnames.append(filename)

        TTM_fname = TTM_fnames[continue_ttm]

        print(TTM_fname)

        Y = np.zeros((1, hp.max_T, hp.n_mels), np.float32)
        L = np.zeros((1, hp.max_N), np.float32)

        align = tf.zeros((1, hp.max_N, hp.max_T), tf.float32)
        max_att_loss = tf.zeros((1, hp.max_T), tf.float32)
        prev_max_attentions = np.zeros((1,), np.int32)

        model_TTM.train_on_batch(x=[L, Y, prev_max_attentions],
                                 y=[Y, Y, align, max_att_loss])

        model_TTM.load_weights(os.path.join(hp.data, "finished_models", TTM_fname))

        gv = int(TTM_fname[11:14]) * 1647
        start_epoch = int(TTM_fname[11:14]) + 1

        print("weights loaded")

    log_loss = pd.DataFrame({
        'loss': np.zeros(0),
        'loss_mels': np.zeros(0),
        'loss_bd1': np.zeros(0),
        'loss_att': np.zeros(0)
    })

    log_loss_append = pd.DataFrame({
        'loss': np.zeros(1),
        'loss_mels': np.zeros(1),
        'loss_bd1': np.zeros(1),
        'loss_att': np.zeros(1)
    })

    for epoch in range(start_epoch, hp.epochs+1):
        # create queue
        df = pd.DataFrame(
            {'fpath': np_ar[:, 0],
             'text_length': np_ar[:, 1],
             'texts': np_ar[:, 2]
             }).sample(n=len(fpaths))
        df['bucket_num'] = np.ceil((df['text_length']+1-minlen) / 20)

        df_queue = pd.DataFrame({
            'fpath': np.zeros(0),
            'text_length': np.zeros(0),
            'texts': np.zeros(0),
            'bucket_num': np.zeros(0),
            'batch_num': np.zeros(0)
        })

        for i in range(1,int(max(df['bucket_num']))+1):
            df_queue_append = df.loc[df['bucket_num'] == int(i)]
            df_queue_append['batch_num'] = np.ceil(np.arange(df_queue_append.shape[0])/hp.B)+1
            df_queue = df_queue.append(df_queue_append)

        queue_rand_order = df_queue.iloc[:,3:].drop_duplicates()
        queue_rand_order = queue_rand_order.sample(n=queue_rand_order.shape[0])
        queue_rand_order['row_num'] = np.arange(queue_rand_order.shape[0])+1

        # iterate over batches
        for n_batch in range(1,int(max(queue_rand_order['row_num']))+1):
            # get batch and bucket num for this iteration
            df_bucket_n = queue_rand_order.loc[queue_rand_order['row_num'] == n_batch].iloc[:,0]
            df_batch_n = queue_rand_order.loc[queue_rand_order['row_num'] == n_batch].iloc[:,1]

            # get filepaths and texts based on batch and bucket num
            fpaths_batch = df_queue.loc[(df_queue['bucket_num'] == int(df_bucket_n)) & (df_queue['batch_num'] == int(df_batch_n))].iloc[:,0].tolist()
            L = df_queue.loc[(df_queue['bucket_num'] == int(df_bucket_n)) & (df_queue['batch_num'] == int(df_batch_n))].iloc[:,2].tolist()

            max_len = 0
            for n in range(len(L)):
                if L[n].shape[0] > max_len:
                    max_len = L[n].shape[0]

            for n in range(len(L)):
                num_paddings = max_len - L[n].shape[0]
                L[n] = np.pad(L[n], [0, num_paddings], mode="constant")

            mels, mags, fnames, max_mel, _ = get_batch(fpaths_batch)

            prev_max_attentions = tf.ones(shape=(len(L),), dtype=tf.int32)
            S = tf.concat((tf.zeros_like(mels[:, :1, :]), mels[:, :-1, :]), 1)
            max_att_loss = tf.zeros((len(L), max_mel), tf.float32)

            # needed for exploratory inference
            batch_lr = learning_rate_decay(hp.lr, gv)

            Ker.set_value(model_TTM.optimizer.learning_rate, batch_lr)
            #model_TTM.optimizer._lr = batch_lr

            loss = model_TTM.train_on_batch(x=[L, S, prev_max_attentions],
                                            y=[mels, mels, align, max_att_loss]); gv += 1

            log_loss_append['loss'].iloc[0] = loss[0]
            log_loss_append['loss_mels'].iloc[0] = loss[1]
            log_loss_append['loss_bd1'].iloc[0] = loss[2]
            log_loss_append['loss_att'].iloc[0] = loss[3]
            log_loss = log_loss.append(log_loss_append)

            if n_batch % 10 == 0:
                print(datetime.datetime.now(),"batch "+str(n_batch)+"/"+str(int(max(queue_rand_order['row_num'])))+": "+
                      str(round(np.mean(log_loss['loss'].iloc[-10:]),4))+" - "+
                      str(round(np.mean(log_loss['loss_mels'].iloc[-10:]),4))+" - "+
                      str(round(np.mean(log_loss['loss_bd1'].iloc[-10:]),4))+" - "+
                      str(round(np.mean(log_loss['loss_att'].iloc[-10:]),5)))

            #if n_batch == int(max(queue_rand_order['row_num'])):
            if n_batch == 1:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                savepath = os.path.join(hp.data, "finished_models", "model_TTM_e" + str(1000+epoch)[1:] + "_t" + timestamp)

                model_TTM.save_weights(savepath+".h5", save_format='h5')
                pd.DataFrame(log_loss).to_csv(os.path.join(hp.data, "outputs", "tfv2_logloss_" + timestamp + ".csv"))
                print("model saved")


def train_SSRN(ssrn_rmean=hp.ssrn_rmean,
               ssrn_rmean_logits=hp.ssrn_rmean_logits,
               continue_training=False,
               continue_ssrn=-1):
    fpaths, text_lengths, texts, maxlen, minlen = load_data()  # list

    print("Texts loaded")

    np_ar = np.transpose(np.array([fpaths, text_lengths, texts]))

    df = pd.DataFrame(
        {'fpath': np_ar[:, 0],
         'text_length': np_ar[:, 1],
         'texts': np_ar[:, 2]
        })

    df = df.sort_values(by=['text_length'])
    df['batch_num'] = np.ceil((np.arange(len(df))+1)/hp.B)

    df_rand_order = pd.DataFrame(
        {'batch_n': np.arange(np.ceil(13100/hp.B))+1}).sample(n=int(np.ceil(13100/hp.B)))

    df_rand_order['row_num'] = np.arange(np.ceil(13100/hp.B))+1

    # model build
    model_SSRN = SSRN()
    print("SSRN Model loaded")

    model_SSRN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.lr,
                                                                 clipvalue=1.,
                                                                 beta_1=hp.beta1,
                                                                 beta_2=hp.beta2,
                                                                 epsilon=hp.epsilon),
                       loss=[C_Reduce_Mean(),
                             C_Reduce_Mean_Logits()],
                       loss_weights=[float(ssrn_rmean),
                                     float(ssrn_rmean_logits)])

    gv = 0
    start_epoch = 1

    if continue_training==True:
        SSRN_fnames = []
        for filename in os.listdir(os.path.join(hp.data, "finished_models")):
            if filename.startswith("model_SSRN"):
                SSRN_fnames.append(filename)

        SSRN_fname = SSRN_fnames[continue_ssrn]

        print(SSRN_fname)

        Y = np.zeros((1, hp.max_T, hp.n_mels), np.float32)
        Z = np.zeros((1, hp.max_T * 4, 1 + hp.n_fft // 2), np.float32)

        model_SSRN.train_on_batch(x=[Y],
                                  y=[Z, Z])

        model_SSRN.load_weights(os.path.join(hp.data, "finished_models", SSRN_fname))

        gv = int(SSRN_fname[12:15]) * 3285
        start_epoch = int(SSRN_fname[12:15]) + 1

        print(gv)

        print("weights loaded")

    log_loss = pd.DataFrame({
        'loss': np.zeros(0),
        'loss_mags': np.zeros(0),
        'loss_bd2': np.zeros(0)
    })

    log_loss_append = pd.DataFrame({
        'loss': np.zeros(1),
        'loss_mags': np.zeros(1),
        'loss_bd2': np.zeros(1)
    })


    for epoch in range(start_epoch, hp.epochs+1):
        # create queue
        df = pd.DataFrame(
            {'fpath': np_ar[:, 0],
             'text_length': np_ar[:, 1],
             'texts': np_ar[:, 2]
             }).sample(n=len(fpaths))
        df['bucket_num'] = np.ceil((df['text_length']+1-minlen) / 20)

        df_queue = pd.DataFrame({
            'fpath': np.zeros(0),
            'text_length': np.zeros(0),
            'texts': np.zeros(0),
            'bucket_num': np.zeros(0),
            'batch_num': np.zeros(0)
        })

        for i in range(1,int(max(df['bucket_num']))+1):
            df_queue_append = df.loc[df['bucket_num'] == int(i)]
            df_queue_append['batch_num'] = np.ceil(np.arange(df_queue_append.shape[0])/hp.B)+1
            df_queue = df_queue.append(df_queue_append)

        queue_rand_order = df_queue.iloc[:,3:].drop_duplicates()
        queue_rand_order = queue_rand_order.sample(n=queue_rand_order.shape[0])
        queue_rand_order['row_num'] = np.arange(queue_rand_order.shape[0])+1

        for n_batch in range(1,int(max(queue_rand_order['row_num']))+1):
            df_bucket_n = queue_rand_order.loc[queue_rand_order['row_num'] == n_batch].iloc[:,0]
            df_batch_n = queue_rand_order.loc[queue_rand_order['row_num'] == n_batch].iloc[:,1]

            # get filepaths and texts based on batch and bucket num
            fpaths_batch = df_queue.loc[(df_queue['bucket_num'] == int(df_bucket_n)) & (df_queue['batch_num'] == int(df_batch_n))].iloc[:,0].tolist()

            mels, mags, fnames, _, _ = get_batch(fpaths_batch)

            batch_lr = learning_rate_decay(hp.lr, gv)

            model_SSRN.optimizer._lr = batch_lr

            loss = model_SSRN.train_on_batch(x=[mels],
                                             y=[mags, mags]); gv += 1

            log_loss_append['loss'].iloc[0] = loss[0]
            log_loss_append['loss_mags'].iloc[0] = loss[1]
            log_loss_append['loss_bd2'].iloc[0] = loss[2]
            log_loss = log_loss.append(log_loss_append)

            if n_batch % 10 == 0:
                print(datetime.datetime.now(),"batch "+str(n_batch)+"/"+str(int(max(queue_rand_order['row_num'])))+": "+
                      str(round(np.mean(log_loss['loss'].iloc[-10:]),4))+" - "+
                      str(round(np.mean(log_loss['loss_mags'].iloc[-10:]),4))+" - "+
                      str(round(np.mean(log_loss['loss_bd2'].iloc[-10:]),4)))

            #if n_batch == int(max(queue_rand_order['row_num'])):
            if n_batch == 1:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                savepath = os.path.join(hp.data, "finished_models","model_SSRN_e" + str(1000+epoch)[1:] + "_t" + timestamp)

                model_SSRN.save_weights(savepath+".h5", save_format='h5')
                pd.DataFrame(log_loss).to_csv(os.path.join(hp.data, "outputs", "tfv2_logloss_SSRN_" + timestamp + ".csv"))
                print("model saved")


for _ in range(1):
    #train_TTM(continue_training=False, continue_ttm=-1)
    train_SSRN(continue_training=False, continue_ssrn=-1)

