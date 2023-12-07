import os
import gc
import joblib
import time
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from utils import timer

from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from sklearn.preprocessing import QuantileTransformer

MLP_NUM_MODELS = 5
LR_START = 1e-6
LR_MAX = 1e-3
LR_MIN = 1e-6
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 50
lr = 0.01

df = pd.read_feather("input/train_processed.f")

y = df['target']
dates = df.date_id
time_id = df['time_id'].values
stock_id = df['stock_id'].values

dropcols = ['time_id','date_id','stock_id','target']
df.drop(dropcols, axis=1, inplace=True)
gc.collect()
print(df.shape)

with timer("Quantile scaling applied to train set"):
    qt = QuantileTransformer(n_quantiles=2000, output_distribution='normal')
    df[:] = qt.fit_transform(df)
    df.fillna(0, inplace=True)
    joblib.dump(qt, 'QuantileTransformer.pkl')

with timer("Reformatted training shape"):
    ntime = 26455
    nstock = 200
    nfeatures = df.shape[1]

    Xtrain = np.zeros((ntime, nstock, nfeatures))
    Ytrain = np.zeros((ntime, nstock))
    Xdate = np.zeros((ntime, nstock))

    Xtrain[time_id, stock_id, :] = df.values
    Ytrain[time_id, stock_id] = y.values
    Xdate[time_id, stock_id] = dates

    Xdate = Xdate[:,0]

    print(Xtrain.shape, Ytrain.shape, Xdate.shape)

    del df
    gc.collect()

with timer('make folds'):
    folds_border = [481 - 45*4, 481 - 45*3, 481 - 45*2, 481 - 45*1]

    folds = []
    for i, border in enumerate(folds_border):
        idx_train = np.where(Xdate < border)[0]
        idx_valid = np.where((border <= Xdate) & (Xdate < border + 45))[0]
        folds.append((idx_train, idx_valid))

        print(f"folds{i}: train={len(idx_train)}, valid={len(idx_valid)}")


def get_model():
    inp = tf.keras.layers.Input(shape=(nstock, nfeatures))

    x = tf.keras.layers.Dropout(0.01, noise_shape=(1, nfeatures))(inp)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(loss='mae', optimizer='adam')
    return model


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr


rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, '-o')
plt.xlabel('Epoch');
plt.ylabel('LR')
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}". \
      format(lr_y[0], max(lr_y), lr_y[-1]))
LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

# Save the model to a file
model_save_path = 'Dan/trained_models'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

with timer('mlp.cv'):
    scores = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        x_train, y_train = Xtrain[train_idx], np.clip(Ytrain[train_idx], -25, 25)
        x_val, y_val = Xtrain[val_idx], Ytrain[val_idx]
        model = get_model()
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=256, verbose=1, validation_data=(x_val, y_val), callbacks=[LR])

        val_pred = model.predict(x_val)[:,:,0]
        score = np.mean(np.abs(y_val - val_pred))
        scores.append(score)
        model_filename = os.path.join(model_save_path, f'mlp_validation_{fold + 1}.h5')
        print(f"Fold {fold} MAE: {score}")
        model.save(model_filename)
        del model
        gc.collect()
    print(f"# overall MAE: {np.mean(np.array(scores))} +/- {np.std(np.array(scores))}")

# overall MAE: 6.1102131139258296 +/- 0.23200097729559557