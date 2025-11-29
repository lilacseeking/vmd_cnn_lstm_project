import numpy as np
import pandas as pd
from vmdpy import VMD
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import lightgbm as lgb
import os

# 创建target文件夹用于保存图像
TARGET_DIR = 'target'
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

def normalize_series(arr):
    arr = np.array(arr).astype(float)
    arr_max = np.max(arr)
    arr_min = np.min(arr)
    if arr_max == arr_min:
        return arr, arr_min, arr_max
    return (arr - arr_min) / (arr_max - arr_min), arr_min, arr_max

def create_dataset(data, win_size=12):
    """
    使用滑动窗口创建监督学习数据集。
    """
    X = []
    Y = []
    for i in range(len(data) - win_size):
        temp_x = data[i:i + win_size]
        temp_y = data[i + win_size]
        X.append(temp_x)
        Y.append(temp_y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

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

# 新增：残差项分析功能
def analyze_residual(residual_imf, imf_index, signal_length):
    """
    分析残差项的各种信息
    
    Parameters:
    residual_imf: 残差IMF分量
    imf_index: IMF索引
    signal_length: 原始信号长度
    """
    print(f"\n=== IMF {imf_index+1} 残差项分析 ===")
    
    # 基本统计信息
    mean_val = np.mean(residual_imf)
    std_val = np.std(residual_imf)
    min_val = np.min(residual_imf)
    max_val = np.max(residual_imf)
    
    print(f"基本统计信息:")
    print(f"  均值: {mean_val:.6f}")
    print(f"  标准差: {std_val:.6f}")
    print(f"  最小值: {min_val:.6f}")
    print(f"  最大值: {max_val:.6f}")
    
    # 绘制残差项图像
    plt.figure(figsize=(12, 6))
    plt.plot(residual_imf)
    plt.title(f'IMF {imf_index+1} 残差项')
    plt.xlabel('时间点')
    plt.ylabel('幅值')
    plt.grid(True)
    # 保存图像
    plt.savefig(os.path.join(TARGET_DIR, f'IMF{imf_index+1}_残差项.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': min_val,
        'length': len(residual_imf)
    }

# 新增：残差项数值计算功能
def compute_residual_operations(residual_imf):
    """
    对残差项进行各种数值计算
    """
    print(f"\n=== 残差项数值计算 ===")
    
    # 计算差分
    diff_residual = np.diff(residual_imf)
    print(f"差分结果统计:")
    print(f"  差分均值: {np.mean(diff_residual):.6f}")
    print(f"  差分标准差: {np.std(diff_residual):.6f}")
    
    # 计算累积和
    cumsum_residual = np.cumsum(residual_imf)
    print(f"累积和最终值: {cumsum_residual[-1]:.6f}")
    
    # 计算移动平均
    window_size = min(10, len(residual_imf)//10)  # 窗口大小为序列长度的1/10或10
    if window_size > 0:
        moving_avg = np.convolve(residual_imf, np.ones(window_size)/window_size, mode='valid')
        print(f"移动平均 (窗口={window_size}):")
        print(f"  移动平均均值: {np.mean(moving_avg):.6f}")
        print(f"  移动平均标准差: {np.std(moving_avg):.6f}")
    
    # 绘制计算结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始残差
    axes[0,0].plot(residual_imf)
    axes[0,0].set_title('原始残差项')
    axes[0,0].grid(True)
    
    # 差分
    axes[0,1].plot(diff_residual)
    axes[0,1].set_title('残差项差分')
    axes[0,1].grid(True)
    
    # 累积和
    axes[1,0].plot(cumsum_residual)
    axes[1,0].set_title('残差项累积和')
    axes[1,0].grid(True)
    
    # 移动平均
    if window_size > 0:
        axes[1,1].plot(moving_avg)
        axes[1,1].set_title(f'残差项移动平均 (窗口={window_size})')
        axes[1,1].grid(True)
    else:
        axes[1,1].text(0.5, 0.5, '数据不足无法计算\n移动平均', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('残差项移动平均')
    
    plt.tight_layout()
    # 保存图像
    plt.savefig(os.path.join(TARGET_DIR, f'IMF{imf_index+1}_残差项数值计算.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 新增：识别趋势IMF（残差项）
def identify_trend_imf(imfs, raw_data):
    """
    识别最能代表原始数据整体趋势的IMF作为残差项
    通过计算每个IMF与原始数据的相关性来判断
    """
    correlations = []
    for i in range(imfs.shape[0]):
        corr = np.corrcoef(imfs[i, :], raw_data)[0, 1]
        correlations.append((i, abs(corr)))
        print(f"IMF {i+1} 与原始数据的相关系数: {corr:.4f}")
    
    # 选择相关系数最大的IMF作为趋势项（残差项）
    trend_imf_index = max(correlations, key=lambda x: x[1])[0]
    print(f"选择 IMF {trend_imf_index+1} 作为残差项（趋势项）")
    
    return trend_imf_index

# 新增：生成影响因子
def generate_influencing_factors(power_data):
    """
    根据原始需求量数据生成7个影响因子
    """
    factors = {
        '电力工程投资': 0.163 * power_data,
        '电力设备购置投资': 0.158 * power_data,
        '原材料成本': 0.15 * power_data,
        '极端气温事件': 0.138 * power_data,
        '暴雨洪涝灾害': 0.135 * power_data,
        '人工成本': 0.131 * power_data,
        '干旱灾害': 0.125 * power_data
    }
    return factors

# 新增：使用影响因子进行预测
def predict_with_influencing_factors(factors, window_size=12, test_size=0.2, create_cnn_lstm_model=None, dataset=None, future_steps=30):
    """
    使用7个影响因子分别进行CNN-LSTM预测
    """
    factor_predictions = {}
    factor_test_values = {}
    factor_future_predictions = {}  # 存储未来预测值
    
    for factor_name, factor_data in factors.items():
        print(f"\n处理影响因子: {factor_name}")
        
        # 数据标准化
        arr_max = np.max(factor_data)
        arr_min = np.min(factor_data)
        if arr_max == arr_min:
            data_normalized = np.zeros_like(factor_data)
        else:
            data_normalized = (factor_data - arr_min) / (arr_max - arr_min)
        
        # 构建数据集
        X, Y = dataset(data_normalized, window_size)
        X = np.expand_dims(X, axis=2)
        
        # 划分训练集和测试集
        train_X, test_X, train_Y, test_Y = train_test_split(
            X, Y, test_size=test_size, shuffle=False
        )
        
        # 创建并训练模型
        model = create_cnn_lstm_model(
            input_shape=(train_X.shape[1], train_X.shape[2]), 
            model_type='simple', 
            use_pooling=True
        )
        
        # 添加回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0, min_lr=1e-6)
        ]
        
        model.fit(
            train_X, train_Y, 
            epochs=18, batch_size=64, 
            validation_split=0.2, shuffle=False, 
            verbose=0,
            callbacks=callbacks
        )
        
        # 预测
        predictions = model.predict(test_X, verbose=0)
        
        # 递归预测未来值
        sequence = test_X[-1]  # 最后一个输入序列
        future_predictions_norm = []
        seq = np.array(sequence).reshape(window_size, 1).astype(np.float32).copy()
        
        for _ in range(future_steps):
            # 调整形状为模型输入
            input_seq = seq.reshape(1, window_size, 1)
            # 预测下一步
            next_pred_norm = model.predict(input_seq, verbose=0)
            # 保存预测结果
            future_predictions_norm.append(float(next_pred_norm[0, 0]))
            # 更新序列
            seq = np.append(seq[1:], np.array([[next_pred_norm[0, 0]]]), axis=0)
        
        # 反归一化
        predictions_denormalized = predictions.flatten() * (arr_max - arr_min) + arr_min
        test_Y_denormalized = test_Y * (arr_max - arr_min) + arr_min
        future_predictions_denormalized = np.array(future_predictions_norm) * (arr_max - arr_min) + arr_min
        
        # 保存结果
        factor_predictions[factor_name] = predictions_denormalized
        factor_test_values[factor_name] = test_Y_denormalized
        factor_future_predictions[factor_name] = future_predictions_denormalized
        
        # 绘制预测结果图（仅预测值）
        plt.figure(figsize=(15, 6), dpi=100)
        plt.plot(predictions_denormalized, color='r', label=f'{factor_name} 预测值')
        plt.title(f'{factor_name} 预测值', fontsize=20)
        plt.grid(True)
        plt.xlabel('时间步 (测试集)', fontsize=18)
        plt.ylabel('数值', fontsize=18)
        plt.legend(fontsize=16)
        # 保存图像
        plt.savefig(os.path.join(TARGET_DIR, f'{factor_name}_预测值.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    return factor_predictions, factor_test_values, factor_future_predictions

# 新增：使用LightGBM进行集成预测
def ensemble_predict_with_lightgbm(imf_predictions, test_size=0.2):
    """
    使用LightGBM对5个IMF的预测结果进行集成
    
    Parameters:
    imf_predictions: 包含5个IMF预测结果的字典
    """
    # 准备数据：将5个IMF的预测结果作为特征
    features = np.column_stack(list(imf_predictions.values()))
    
    # 目标变量：5个IMF预测结果的真实值之和
    target = np.sum(list(imf_predictions.values()), axis=0)
    
    # 划分训练集和测试集
    split_index = int(len(features) * (1 - test_size))
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = target[:split_index], target[split_index:]
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # LightGBM参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1
    }
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # 预测
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # 评估
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n=== LightGBM 集成模型评估结果 ===")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    return model, y_pred, y_test