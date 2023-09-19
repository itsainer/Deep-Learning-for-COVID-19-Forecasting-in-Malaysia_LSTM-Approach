
#%%
"""
Task: Creating a deep learning model using LSTM neural network 
to predict new COVID-19 cases in Malaysia using the past 30 days of number of cases.
"""
#%%
#IMPORT PACKAGES
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from ts_window_helper import WindowGenerator

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# %%
# LOAD DATA
df = pd.read_csv('cases_malaysia_covid.csv')
# %%
# REMOVE DATE COLUMN, SEPARATE
date = pd.to_datetime(df.pop('date'), format='%d/%m/%Y')
# %%
# DATA CLEANING
df.info()
# 1.Change dtype of cases_new (object -> int)
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')
# 2.Drop all columns after cases_active
df.drop(df.loc[:, 'cases_cluster':'cluster_workplace'].columns, axis=1, inplace=True)
# 3.Fill missing values with 0
df.fillna(0, inplace=True)
# 4.Drop duplicate values
df.drop_duplicates(inplace=True)
#%%
# Reduce no of date series -> 770
index_remove = range(0,10)
date = date[~date.index.isin(index_remove)]
# %%
#Plotting graphs to inspect any trends
plot_cols = ['cases_new','cases_import','cases_active','cases_recovered']
plot_features = df[plot_cols]
plot_features.index = date
_ = plot_features.plot(subplots=True)
# %%
#Inspect basic statistics from dataset
df.describe().transpose()
# %%
# DATA SPLITTING

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

#check no of columns
num_features = df.shape[1]
# %%
# Data Normalization

#use mean and stddev from training data
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
# %%
#Inspect distribution of features after normalization
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
# %%
# MODEL DEVELOPMENT

#Create data window for Single-Step LSTM
single_window = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df, 
                              input_width=30, label_width=30, shift=1, label_columns=['cases_new'])

#Plot window
single_window.plot('cases_new')
# %%
#Visualize by Tensorboard
base_log_path = r"tensorboard_logs\single-step"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = tf.keras.callbacks.TensorBoard(log_path)
#%%
# BUILD MODEL
# RNN :SINGLE-STEP LSTM -> SINGLE OUTPUT

single_lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Add Dropout layer
    tf.keras.layers.Dropout(0.4),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)])
# %%
#Define MAPE function
def mape(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])  # Flatten the true values
    y_pred = tf.reshape(y_pred, [-1])  # Flatten the predicted values
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))

MAX_EPOCHS = 20
#Define function to compile and fit model
def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError(), mape])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[tb, early_stopping])
  return history
#%%
# COMPILE AND FIT SINGLE-STEP LSTM
history = compile_and_fit(single_lstm_model, single_window, patience=2)
# %%
#Plot model labels and predictions
single_window.plot(plot_col='cases_new', model=single_lstm_model)
#Display model summary
single_lstm_model.summary()
#Display model structure
tf.keras.utils.plot_model(single_lstm_model)
#Plot model performance
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()
# %%
# BUILD MODEL
# RNN :MULTI-STEP LSTM

#Create data window
OUT_STEPS = 30
multi_window = WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,input_width=30,
                               label_width=OUT_STEPS, shift=OUT_STEPS)

#Plot window
multi_window.plot(plot_col='cases_new')
# %%
#Create model
for example_inputs, example_labels in multi_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
multi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([OUT_STEPS, example_labels.shape[-1]])
])
#%%
#Visualize by Tensorboard
base_log_path = r"tensorboard_logs\multi-steps"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = tf.keras.callbacks.TensorBoard(log_path)
# %%
# COMPILE AND FIT MODEL
history = compile_and_fit(multi_lstm_model, multi_window, patience=2)
#Plot model labels and predictions
multi_window.plot('cases_new',multi_lstm_model)
#%%
#Display model summary
multi_lstm_model.summary()
#Display model structure
tf.keras.utils.plot_model(multi_lstm_model)
#Display model performance
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()
#%%
#SAVE MODELS
single_lstm_model.save(os.path.join('models', 'single_lstm_model.h5'))
multi_lstm_model.save(os.path.join('models', 'multi_lstm_model.h5'))
# %%
# Export df
# Define the file path
file_path = 'df_covid_malaysia.csv' 

# Export the cleaned DataFrame to a CSV file
df.to_csv(file_path, index=False)

print(f'dataframe (df) has been saved to {file_path}')
# %%
