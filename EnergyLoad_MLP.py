import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import time
import gc

import data_generators
import nn_models
from nn_models import *


import os




def model_train(train_path, m_path_prefix, model_id, in_size, frc_horizon, layers_num, layers_size, batch_size, steps):
    # Training step implementation
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)            
            # loss_f = tf.keras.losses.MeanSquaredError()
            loss_f = tf.keras.losses.MeanAbsoluteError()
            loss_value = loss_f(y, logits)
                
        grads = tape.gradient(loss_value, model.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

    # Training data
    train_file = np.load(train_path)
    train_set = data_generators.ds_generator_historical(train_file, batch_size, frc_horizon)
    train_set = iter(train_set)

    # Model
    # if layers_num==3:
    #     model = nn_models.shallow_mlp_model(in_size, layers_num, layers_size, frc_horizon)
    # elif layers_num==1:
    #     model = nn_models.shallow_mlp_model(in_size, layers_num, layers_size, frc_horizon)
    # elif layers_num==10:
    #     model = nn_models.deep_residual_mlp_model(in_size, layers_size, frc_horizon)
    # else:
    #     model = nn_models.deep_residual_mlp_model(in_size, layers_size, frc_horizon)

    
    #model = basic_attention_model(in_size, 6, 64, frc_horizon)                    #in_size, num_heads, att_dim, fh
    #model = mlp_attention_model(in_size, 64, 6, frc_horizon)                      #in_size, att_dim, num_heads, fh
    #model = positional_attention_model(in_size, 64, 1, 6, frc_horizon)            #in_size, att_dim, dim, num_heads, fh
    #model = mlp_positional_attention_model(in_size, 64, 1, 6, frc_horizon)        #in_size, att_dim, dim, num_heads, fh
    model = positional_mlp_attention_model(in_size, 64, 1, 6, frc_horizon)        #in_size, att_dim, dim, num_heads, fh
    #model = lstm_attention_model(in_size, 6, 64, frc_horizon)                     #in_size, num_heads, att_dim, fh
    #model = mlp_lstm_attention_model(in_size, 6, 64, frc_horizon)                 #in_size, num_heads, att_dim, fh
    
    # Training parameters
    number_of_steps = steps
    init_lr = 0.001

    opt = tf.optimizers.experimental.AdamW(learning_rate=init_lr)

    # Training loop
    print(f'\nStarted training...')
    print(m_path_prefix)
    for step in range(number_of_steps):
        x_batch_train, y_batch_train = train_set.get_next()
        loss_value = train_step(x_batch_train, y_batch_train)
        if step % 1000 == 0:
            print(f'Training loss at step {step}: {float(loss_value)}')

    # Save model
    m_path = m_path_prefix + str(model_id) + '.h5'
    model.save(m_path)

    # Clear memory
    tf.keras.backend.clear_session()
    gc.collect()
    return 1

def generate_forecasts(test_path, m_path_prefix, number_of_models, input_size, fh, save_path_prefix):
    # Load test data
    df_test = np.load(test_path)
    x_test = df_test[:, -input_size:].astype(float)

    y_hat_all = list([])
    for i in range(number_of_models): #number_of_models
        # Load model and forecast
        print(f'Generating forecasts from network: {(i + 1)}')
        # Load Model
        pm = m_path_prefix + str(i) + '.h5'
        model = tf.keras.models.load_model(pm, compile=False) #custom_objects={'MultiHeadAttention': MultiHeadAttention(2, 36, 36, 64)}
        # Predict
        y_hat_temp = model(x_test, training=False)
        y_hat_all.append(y_hat_temp)


    # Combine forecasts
    y_hat_all = np.asarray(y_hat_all)
    y_hat_all = np.median(y_hat_all, axis=0)

    # Scale back forecasts
    # x_min = df_test[:, 3:4].astype(float)
    # x_max = df_test[:, 4:5].astype(float)
    # y_hat_all = (y_hat_all * (x_max - x_min)) + x_min

    x_mean = df_test[:, 3:4].astype(float)
    x_std = df_test[:, 4:5].astype(float)
    y_hat_all = (y_hat_all * x_std) + x_mean
    
    # Save ensemble forecasts
    df_predictions = pd.DataFrame({'Country': df_test[:, 1], 'Date': df_test[:, 2]})
    df_predictions = pd.concat([df_predictions, pd.DataFrame(y_hat_all)], axis=1)

    out_path = save_path_prefix + '.csv'
    df_predictions.to_csv(out_path, index=None)
    print(f'Saved forecasts. File shape: {df_predictions.shape}')

# Experiment / Model parameters
s_rate = 12 # sampling rate of the training data: every 24h, 12h, 1h
f_h = 36 # forecasting horizon
ins = 168 # number of past observations
lnum = 7 # number of dense layers
lsize = 512 # size of dense layers
lossf = 'mae' # training loss
batch = 512 # batch size #512 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
number_of_models = 5
steps=20000
train_path = f'data/train_{s_rate}h_in{ins}_counorm.npy'
test_path = f'data/x_test_in{ins}_counorm.npy'

# Train models
for nm in range(number_of_models):
    print(nm)
    mpref = f'models/attention_{steps}_steps_'
    start_time = time.time()
    ret_val = model_train(train_path, mpref, nm, ins, f_h, lnum, lsize, batch, steps)
    end_time = time.time()
    training_time=end_time-start_time
    print("Training time: " + str(training_time) + " seconds, this is "+ str(training_time/60) + " minutes.")
    

# Generate forecasts
mpref = f'models/attention_{steps}_steps_'
save_pref = f'forecasts/mlp_counorm_in{ins}_l{lnum}_{lsize}_{lossf}_{s_rate}h'
#generate_forecasts(test_path, mpref, number_of_models, ins, f_h, save_pref)
    