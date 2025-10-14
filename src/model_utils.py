import numpy as np
import pandas as pd
from vmdpy import VMD
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense

def normalize_series(arr):
    arr = np.array(arr).astype(float)
    arr_max = np.max(arr)
    arr_min = np.min(arr)
    if arr_max == arr_min:
        return arr, arr_min, arr_max
    return (arr - arr_min) / (arr_max - arr_min), arr_min, arr_max

def create_dataset(data, win_size=12):
    X, Y = [], []
    for i in range(len(data) - win_size):
        X.append(data[i:i + win_size])
        Y.append(data[i + win_size])
    return np.asarray(X), np.asarray(Y)

def build_cnn_lstm(input_shape, lstm_units=64):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_units))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam')
    return model

def vmd_decompose(signal, K=5, alpha=1300, tau=0., DC=0, init=1, tol=1e-7):
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    return u

def evaluate_and_print(test_y, y_pred, arr_min, arr_max, label=''):
    y_pred_flat = np.array([i for arr in y_pred for i in arr])
    test_y_orig = test_y * (arr_max - arr_min) + arr_min
    pred_orig = y_pred_flat * (arr_max - arr_min) + arr_min
    mse = metrics.mean_squared_error(test_y_orig, pred_orig)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(test_y_orig, pred_orig)
    r2 = r2_score(test_y_orig, pred_orig)
    print(f"{label} MSE:{mse:.6f} RMSE:{rmse:.6f} MAE:{mae:.6f} R2:{r2:.6f}")
    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2)
