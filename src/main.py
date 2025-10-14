import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_utils import normalize_series, create_dataset, build_cnn_lstm, vmd_decompose, evaluate_and_print

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CSV_FILE = os.path.join(DATA_PATH, 'sample_data.csv')

os.makedirs(DATA_PATH, exist_ok=True)

if not os.path.exists(CSV_FILE):
    print('没有找到 data/sample_data.csv，生成合成示例数据...')
    rng = pd.date_range(start='2020-01-01', periods=500, freq='D')
    t = np.arange(len(rng))
    series = 50 + 0.02 * t + 5 * np.sin(2 * np.pi * t / 30) + 2 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 0.8, size=len(t))
    df = pd.DataFrame({'date': rng, 'value': series})
    df.to_csv(CSV_FILE, index=False)

print('加载数据...')
df = pd.read_csv(CSV_FILE)
series = df['value'].values if 'value' in df.columns else df.iloc[:, 1].values

print('进行 VMD 分解...')
u = vmd_decompose(series, K=5)

imf_dfs = {f'imf_{i+1}': pd.DataFrame({'Value': u[i]}) for i in range(u.shape[0])}
win = 12
all_y_preds, all_series_forecast = [], []

for idx in range(u.shape[0]):
    key = f'imf_{idx+1}'
    arr = imf_dfs[key]['Value'].values.astype(float)
    data_norm, arr_min, arr_max = normalize_series(arr)
    X, Y = create_dataset(data_norm, win_size=win)
    X = np.expand_dims(X, axis=2)
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, shuffle=False)
    model = build_cnn_lstm((train_x.shape[1], train_x.shape[2]), lstm_units=64)
    model.fit(train_x, train_y, epochs=5, batch_size=32, validation_split=0.2, shuffle=False, verbose=0)
    y_pred = model.predict(test_x)
    all_y_preds.append((y_pred, arr_min, arr_max))
    evaluate_and_print(test_y, y_pred, arr_min, arr_max, label=key)

print('完成。')
