import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from vmdpy import VMD
import os

# 设置中文显示和绘图样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 修正：设置默认的中文字体，例如 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['axes.titlesize'] = 20  # 标题字体大小
plt.rcParams['axes.labelsize'] = 18  # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 16  # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 16  # 图例字体大小

WINDOW_SIZE = 12  # 时间窗口大小 (Lookback period)
FUTURE_STEPS = 30  # 未来预测步长
TEST_SIZE = 0.2

# 创建target文件夹用于保存图像
TARGET_DIR = 'target'
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)


# --- 1. 数据生成（模拟文章中的 '数据.xlsx' 文件） ---
def generate_synthetic_data(file_name='数据.xlsx'):
    """生成模拟的每日功率波动数据并保存为 Excel 文件。"""
    np.random.seed(42)
    days = 2000
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

    # 长期趋势
    trend = np.linspace(500, 600, days)
    # 季节性 (周期约 90 天)
    seasonality_1 = 50 * np.sin(np.linspace(0, 4 * np.pi, days))
    seasonality_2 = 20 * np.sin(np.linspace(0, 20 * np.pi, days))
    # 噪声
    noise = np.random.normal(0, 5, days)

    # 最终序列
    power_data = trend + seasonality_1 + seasonality_2 + noise
    power_data[power_data < 0] = 0  # 确保功率非负

    df = pd.DataFrame({
        '数据时间': dates,
        '需求量(个)': power_data

        # '电力工程投资':0.163 * power_data,
        # '电力设备购置投资':0.158 * power_data,
        # '原材料成本':0.15 * power_data,
        # '极端气温事件':0.138 * power_data,
        # '暴雨洪涝灾害':0.135 * power_data,
        # '人工成本':0.131 * power_data,
        # '干旱灾害':0.125 * power_data
        # '需求量(个)(归一化)': power_data,
        # '电力工程投资(归一化)': 0.85 * power_data,
        # '电力设备购置投资(归一化)': 0.82 * power_data,
        # '原材料成本(归一化)': 0.78 * power_data,
        # '极端气温事件(归一化)': 0.72 * power_data,
        # '暴雨洪涝灾害(归一化)': 0.70 * power_data,
        # '人工成本(归一化)': 0.68 * power_data,
        # '干旱灾害': 0.65 * power_data,
    })

    df.set_index('数据时间', inplace=True)
    df.to_excel(file_name)
    print(f"已生成模拟数据文件: {file_name}")
    return df


# --- 2. 核心辅助函数定义 ---

def dataset(data, win_size=WINDOW_SIZE):
    """
    使用滑动窗口创建监督学习数据集。
    """
    X = []  # 修正：列表初始化
    Y = []  # 修正：列表初始化
    for i in range(len(data) - win_size):
        temp_x = data[i:i + win_size]
        temp_y = data[i + win_size]
        X.append(temp_x)
        Y.append(temp_y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


def create_cnn_lstm_model(input_shape, model_type='simple', use_pooling=True):
    """
    创建 CNN-LSTM 模型。
    参数:
      - input_shape: (win, channels)
      - model_type: 'simple' 或 'deep'
      - use_pooling: 是否在 Conv 后使用 MaxPooling（高频 IMF 建议禁用）
    """
    model = Sequential()

    # 第一部分：Conv1D 和可选的 MaxPooling1D
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    if use_pooling:
        model.add(MaxPooling1D(pool_size=2))

    if model_type == 'deep':  # 深层结构
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(64, activation='relu'))
        # 使用线性激活函数进行回归预测
        model.add(Dense(1, activation='linear'))
    else:  # 简单结构
        model.add(LSTM(256))
        model.add(Dense(1, activation='linear'))

    # 使用 Huber 损失提高对异常值/噪声的鲁棒性
    model.compile(loss=Huber(), optimizer='adam')
    return model


def predict_future_values(model, sequence, arr_min, arr_max, future_steps=FUTURE_STEPS):
    """
    递归预测未来多步值 (Iterated Forecasting)，并处理归一化/反归一化。

    Args:
        model: 训练好的 Keras 模型。
        sequence: 归一化后的最后一个输入序列 (WINDOW_SIZE, 1) 或 (WINDOW_SIZE,)
        arr_min, arr_max: 当前序列的原始最小值和最大值，用于正确进行反归一化和再归一化。
    """
    prediction_results_norm = []  # 修正：列表初始化

    # 确保输入序列是 (WINDOW_SIZE, 1) 形状
    seq = np.array(sequence).reshape(WINDOW_SIZE, 1).astype(np.float32).copy()

    for _ in range(future_steps):
        # 调整形状为模型输入 (1, WINDOW_SIZE, 1)
        input_seq = seq.reshape(1, WINDOW_SIZE, 1)

        # 预测下一步 (归一化结果，shape (1,1))
        next_day_prediction_norm = model.predict(input_seq, verbose=0)

        # 将归一化预测结果加入列表（保留标量）
        prediction_results_norm.append(float(next_day_prediction_norm[0, 0]))

        # --- 反归一化并重新归一化以维持数值稳定性 ---
        denorm_pred = float(next_day_prediction_norm[0, 0]) * (arr_max - arr_min) + arr_min
        # 防止除以零
        if arr_max == arr_min:
            renorm_pred = 0.0
        else:
            renorm_pred = (denorm_pred - arr_min) / (arr_max - arr_min)

        # 更新序列：舍弃第一个时间步，将再归一化结果加入序列末尾
        seq = np.append(seq[1:], np.array([[renorm_pred]]), axis=0)

    # 最终反归一化整个预测序列 (用于绘图和最终重构)
    series_future_denorm = np.array(prediction_results_norm).flatten() * (arr_max - arr_min) + arr_min
    return series_future_denorm


def evaluate_and_plot(test_y_norm, y_pred_norm, series_future_denorm, arr_min, arr_max, title_prefix="原始数据"):
    """
    评估模型性能并将实际值、预测值和未来预测值进行可视化。
    """
    # 反归一化测试集实际值和预测值
    test_y_denorm = test_y_norm * (arr_max - arr_min) + arr_min
    y_pred_denorm = np.array(y_pred_norm).flatten() * (arr_max - arr_min) + arr_min

    # 评估指标
    mse = metrics.mean_squared_error(test_y_denorm, y_pred_denorm)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(test_y_denorm, y_pred_denorm)
    r2 = r2_score(test_y_denorm, y_pred_denorm)

    print("-" * 50)
    print(f"{title_prefix} 模型评估结果 (测试集, 反归一化):")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"均方根误差(RMSE): {rmse:.4f}")
    print(f"平均绝对误差(MAE): {mae:.4f}")
    print(f"拟合优度(R^2): {r2:.4f}")
    print("-" * 50)

    # 绘图
    plt.figure(figsize=(15, 6), dpi=100)

    # 实际值 (测试集)
    plt.plot(test_y_denorm, color='c', label=f'{title_prefix} 实际波动曲线')

    # 预测值 (测试集)
    plt.plot(y_pred_denorm, color='r', label=f'{title_prefix} 预测波动曲线')

    # 未来预测值
    start_index = len(y_pred_denorm)
    end_index = start_index + len(series_future_denorm)
    plt.plot(range(start_index, end_index), series_future_denorm, color='b',
             label=f'{title_prefix} 向后预测 {FUTURE_STEPS} 天')

    plt.title(f'{title_prefix} 实际与预测波动比对图', fontsize=20)
    plt.grid(True)
    plt.xlabel('时间步 (测试集)', fontsize=18)
    plt.ylabel('幅度', fontsize=18)
    plt.legend(fontsize=16)
    # 保存图像
    filename = f'{title_prefix}_实际与预测波动比对图.png'
    plt.savefig(os.path.join(TARGET_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()

    # 返回反归一化后的未来预测值和测试集预测值，用于最终重构
    return series_future_denorm, y_pred_denorm


# --- 3. 主流程执行 ---

if __name__ == '__main__':
    # 1. 数据加载与可视化
    data_file = '数据.xlsx'
    if not os.path.exists(data_file):
        df = generate_synthetic_data(data_file)
    else:
        df = pd.read_excel(data_file, index_col=0, parse_dates=True)

    raw_data = df['需求量(个)']

    plt.figure(figsize=(15, 5))
    plt.plot(raw_data, label='原始数据', color='r', alpha=0.7)
    plt.title('原始时间序列图', fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    # 保存图像
    plt.savefig(os.path.join(TARGET_DIR, '原始时间序列图.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # --- 2. 基线模型：CNN-LSTM (对原始数据直接预测) ---
    print("\n" + "=" * 60)
    print("--- 2. 基线 CNN-LSTM 模型 (原始数据) ---")
    print("=" * 60)

    # 原始数据 0-1 标准化 (用于基线模型和最终重构的反归一化)
    arr_max_raw = np.max(np.array(raw_data))
    arr_min_raw = np.min(np.array(raw_data))
    if arr_max_raw == arr_min_raw:
        # 保护性处理（几乎不可能，但防止除零）
        data_bz = np.zeros_like(np.array(raw_data))
    else:
        data_bz = (np.array(raw_data) - arr_min_raw) / (arr_max_raw - arr_min_raw)
    data_bz = data_bz.ravel()

    # 制作数据集
    data_x, data_y = dataset(data_bz, WINDOW_SIZE)
    # 扩展维度为 Conv1D (样本数, 窗口大小, 1)
    data_x = np.expand_dims(data_x, axis=2)

    # 划分训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=TEST_SIZE, shuffle=False)

    # 模型建立与训练 (simple 结构, epochs=18)
    # baseline 使用 pooling，因为原始序列包含低频成分
    cnn_lstm_baseline = create_cnn_lstm_model(input_shape=(train_x.shape[1], train_x.shape[2]), model_type='simple', use_pooling=True)
    # 基线训练回调
    callbacks_base = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0, min_lr=1e-6)
    ]
    history_baseline = cnn_lstm_baseline.fit(
        train_x, train_y, epochs=18, batch_size=64, validation_split=0.2, shuffle=False, verbose=0, callbacks=callbacks_base
    )

    # 绘制训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(history_baseline.history['loss'])
    plt.plot(history_baseline.history['val_loss'], c='r')
    plt.title("基线 CNN-LSTM 模型训练历史", fontsize=20)
    plt.legend(['loss', 'val_loss'], fontsize=16)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    # 保存图像
    plt.savefig(os.path.join(TARGET_DIR, '基线_CNN-LSTM_模型训练历史.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 预测与评估
    y_pred_baseline_norm = cnn_lstm_baseline.predict(test_x, verbose=0)
    sequence_baseline = test_x[-1]  # 最后一个输入序列

    # 基线模型使用原始数据的 Min/Max 进行递归预测的归一化管理
    future_predictions_baseline = predict_future_values(cnn_lstm_baseline, sequence_baseline, arr_min_raw, arr_max_raw, FUTURE_STEPS)
    future_predictions_baseline, y_pred_baseline_denorm = evaluate_and_plot(
        test_y, y_pred_baseline_norm, future_predictions_baseline, arr_min_raw, arr_max_raw, "基线 CNN-LSTM (原始数据)"
    )

    # --- 3. VMD 分解时间序列 ---
    print("\n" + "=" * 60)
    print("--- 3. VMD 分解时间序列 ---")
    print("=" * 60)

    # VMD 参数
    alpha = 1300
    tau = 0.
    K = 5
    DC = 0
    init = 1
    tol = 1e-7

    # 运行 VMD（传入原始一维数组）
    u, u_hat, omega = VMD(raw_data.values, alpha, tau, K, DC, init, tol)

    # 可视化分解后的 K 个 IMF
    imf_dataframes = {}
    plt.figure(figsize=(15, 12))
    for i in range(K):
        imf_name = 'imf_{}'.format(i + 1)
        imf_dataframes[imf_name] = pd.DataFrame({'Value': u[i, :], 'DataTime': raw_data.index})
        imf_dataframes[imf_name].set_index('DataTime', inplace=True)

        plt.subplot(K, 1, i + 1)
        plt.plot(u[i, :])
        plt.title(f'IMF {i + 1} (K={K})', fontsize=18)
        plt.ylabel('幅度', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(TARGET_DIR, 'VMD分解后的K个IMF.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 识别趋势IMF（残差项）
    from model_utils import identify_trend_imf
    trend_imf_index = identify_trend_imf(u, raw_data.values)
    
    # --- 4. VMD-CNN-LSTM (对每个 IMF 进行预测) ---
    print("\n" + "=" * 60)
    print("--- 4. VMD-CNN-LSTM (对每个 IMF 进行预测) ---")
    print("=" * 60)

    # 初始化收集容器
    all_imf_future_predictions = []  # 修正：列表初始化
    all_imf_test_predictions = []  # 修正：列表初始化
    all_imf_predictions_dict = {}  # 用于LightGBM集成的预测结果

    # 计算全局 std 用作阈值参考
    global_std = np.std(raw_data.values)
    hf_threshold = global_std * 0.35  # 高频阈值（可调）

    for i in range(K):
        imf_name = f'imf_{i + 1}'
        imf_data = imf_dataframes[imf_name]['Value']
        print(f"\n--- 开始处理 {imf_name} ---")

        # 如果是趋势IMF（残差项），采用特殊的处理方式
        if i == trend_imf_index:
            print(f"将 {imf_name} 作为残差项进行特殊处理...")
            
            # 生成7个影响因子
            from model_utils import generate_influencing_factors, predict_with_influencing_factors
            factors = generate_influencing_factors(raw_data.values)
            
            # 使用影响因子分别进行预测
            factor_predictions, factor_test_values, factor_future_predictions = predict_with_influencing_factors(
                factors, 
                WINDOW_SIZE, 
                TEST_SIZE, 
                create_cnn_lstm_model, 
                dataset,
                FUTURE_STEPS
            )
            
            # 将7个影响因子的预测结果求和作为残差项的预测结果
            summed_predictions = np.sum(list(factor_predictions.values()), axis=0)
            summed_test_values = np.sum(list(factor_test_values.values()), axis=0)
            summed_future_predictions = np.sum(list(factor_future_predictions.values()), axis=0)
            
            # 获取测试集的实际值（用于绘图）
            # 因为我们是对各个影响因子分别预测然后求和，所以需要获取一个影响因子的测试集真实值作为基准
            base_test_values = list(factor_test_values.values())[0] 
            
            # 计算残差项预测结果的范围，用于绘图
            all_values = np.concatenate([summed_test_values, summed_predictions, summed_future_predictions])
            res_min = np.min(all_values)
            res_max = np.max(all_values)
            margin = (res_max - res_min) * 0.1  # 添加10%的边距
            
            # 绘制残差项的最终预测结果（与实际值对比，包含未来预测）
            plt.figure(figsize=(15, 6), dpi=100)
            
            # 实际值 (测试集)
            plt.plot(summed_test_values, color='c', label=f'{imf_name} 实际波动曲线')
            
            # 预测值 (测试集)
            plt.plot(summed_predictions, color='r', label=f'{imf_name} 预测波动曲线')
            
            # 未来预测值
            start_index = len(summed_predictions)
            end_index = start_index + len(summed_future_predictions)
            plt.plot(range(start_index, end_index), summed_future_predictions, color='b',
                     label=f'{imf_name} 向后预测 {FUTURE_STEPS} 天')
            
            plt.title(f'{imf_name} 实际与预测波动比对图 (基于影响因子)', fontsize=20)
            plt.grid(True)
            plt.xlabel('时间步 (测试集)', fontsize=18)
            plt.ylabel('幅度', fontsize=18)
            plt.ylim(res_min - margin, res_max + margin)  # 根据数据的实际范围设置y轴
            plt.legend(fontsize=16)
            # 保存图像
            plt.savefig(os.path.join(TARGET_DIR, f'{imf_name}_实际与预测波动比对图_基于影响因子.png'), dpi=300, bbox_inches='tight')
            plt.show()

            # 保存结果
            all_imf_future_predictions.append(summed_future_predictions)
            all_imf_test_predictions.append(summed_predictions)
            all_imf_predictions_dict[imf_name] = summed_predictions
            
            print(f"残差项 ({imf_name}) 预测完成")
            continue

        # 其他IMF的标准处理流程
        # 0-1 标准化 (独立缩放：使用当前 IMF 的最大/最小值)
        imf_arr = np.array(imf_data)
        arr_max_imf = np.max(imf_arr)
        arr_min_imf = np.min(imf_arr)
        if arr_max_imf == arr_min_imf:
            data_bz_imf = np.zeros_like(imf_arr)
        else:
            data_bz_imf = (imf_arr - arr_min_imf) / (arr_max_imf - arr_min_imf)
        data_bz_imf = data_bz_imf.ravel()

        # 制作数据集
        data_x_imf, data_y_imf = dataset(data_bz_imf, WINDOW_SIZE)
        data_x_imf = np.expand_dims(data_x_imf, axis=2)

        # 划分训练集和测试集
        train_x_imf, test_x_imf, train_y_imf, test_y_imf = train_test_split(
            data_x_imf, data_y_imf, test_size=TEST_SIZE, shuffle=False
        )

        # 判断是否为高频 IMF：若 std 很小，相对全局阈值则认为是高频（需保留细节）
        imf_std = np.std(imf_arr)
        use_pooling = True
        if imf_std < hf_threshold:
            use_pooling = False  # 高频去掉 pooling 以保留振幅细节

        # 模型选择与训练
        model_type = 'simple'
        epochs = 18
        if imf_name == 'imf_2':  # 假设 IMF2 使用更深层模型和更多迭代
            model_type = 'deep'
            epochs = 50

        cnn_lstm_imf = create_cnn_lstm_model(
            input_shape=(train_x_imf.shape[1], train_x_imf.shape[2]),
            model_type=model_type,
            use_pooling=use_pooling
        )

        # 训练过程（加入 EarlyStopping/ReduceLROnPlateau）
        callbacks_imf = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0, min_lr=1e-6)
        ]
        history_imf = cnn_lstm_imf.fit(
            train_x_imf, train_y_imf, epochs=epochs, batch_size=64,
            validation_split=0.2, shuffle=False, verbose=0, callbacks=callbacks_imf
        )

        # 绘制训练损失
        plt.figure(figsize=(10, 5))
        plt.plot(history_imf.history['loss'])
        plt.plot(history_imf.history['val_loss'], c='r')
        plt.title(f"{imf_name} CNN-LSTM 模型训练历史 (Epochs: {epochs})", fontsize=20)
        plt.legend(['loss', 'val_loss'], fontsize=16)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        # 保存图像
        plt.savefig(os.path.join(TARGET_DIR, f'{imf_name}_CNN-LSTM_模型训练历史.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 预测与评估
        y_pred_imf_norm = cnn_lstm_imf.predict(test_x_imf, verbose=0)
        sequence_imf = test_x_imf[-1]  # 最后一个输入序列 (shape: (WINDOW_SIZE,1))

        # 递归预测 (返回反归一化结果)
        series_future_denorm = predict_future_values(cnn_lstm_imf, sequence_imf, arr_min_imf, arr_max_imf, FUTURE_STEPS)

        # 评估与绘图 (返回反归一化结果)
        series_future_denorm, y_pred_denorm_imf = evaluate_and_plot(
            test_y_imf, y_pred_imf_norm, series_future_denorm,
            arr_min_imf, arr_max_imf, imf_name
        )

        # 收集反归一化后的未来预测值和测试集预测值
        all_imf_future_predictions.append(series_future_denorm)
        all_imf_test_predictions.append(y_pred_denorm_imf)
        all_imf_predictions_dict[imf_name] = y_pred_denorm_imf

    # --- 5. 使用LightGBM进行集成预测 ---
    print("\n" + "=" * 60)
    print("--- 5. 使用LightGBM进行集成预测 ---")
    print("=" * 60)
    
    from model_utils import ensemble_predict_with_lightgbm
    lgb_model, lgb_pred, lgb_test = ensemble_predict_with_lightgbm(all_imf_predictions_dict)

    # --- 6. 最终重构与对比 ---
    print("\n" + "=" * 60)
    print("--- 6. 最终 VMD-CNN-LSTM 重构与对比 ---")
    print("=" * 60)

    # 预测重构 (将所有 IMF 的预测值相加)
    # 注意：各 IMF 的 future 长度应相同，test_pred 长度应相同
    reconstructed_future_denorm = np.sum(np.array(all_imf_future_predictions), axis=0)
    reconstructed_pred_test_denorm = np.sum(np.array(all_imf_test_predictions), axis=0)

    # 提取原始数据的测试集实际值（使用基线分割的 test_y）
    data_y_denorm = test_y * (arr_max_raw - arr_min_raw) + arr_min_raw

    # 评估最终重构结果
    mse_vmd = metrics.mean_squared_error(data_y_denorm, reconstructed_pred_test_denorm)
    rmse_vmd = np.sqrt(mse_vmd)
    mae_vmd = metrics.mean_absolute_error(data_y_denorm, reconstructed_pred_test_denorm)
    r2_vmd = r2_score(data_y_denorm, reconstructed_pred_test_denorm)

    print("\n" + "=" * 60)
    print("VMD-CNN-LSTM 整体模型评估结果 (测试集重构):")
    print(f"均方误差(MSE): {mse_vmd:.4f}")
    print(f"均方根误差(RMSE): {rmse_vmd:.4f}")
    print(f"平均绝对误差(MAE): {mae_vmd:.4f}")
    print(f"拟合优度(R^2): {r2_vmd:.4f}")
    print("=" * 60)

    # 最终结果对比图
    plt.figure(figsize=(15, 6), dpi=100)

    plt.plot(data_y_denorm, color='c', label='原始数据 (实际值)')
    plt.plot(reconstructed_pred_test_denorm, color='r', label='VMD-CNN-LSTM (重构预测值)')

    start_index = len(reconstructed_pred_test_denorm)
    end_index = start_index + len(reconstructed_future_denorm)
    plt.plot(range(start_index, end_index), reconstructed_future_denorm, color='b',
             label=f'VMD-CNN-LSTM 向后预测 {FUTURE_STEPS} 天')

    # 根据数据的实际范围设置y轴，而不是从0开始
    all_values = np.concatenate([data_y_denorm, reconstructed_pred_test_denorm, reconstructed_future_denorm])
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_range = y_max - y_min
    margin = y_range * 0.1  # 添加10%的边距
    plt.ylim(y_min - margin, y_max + margin)

    plt.title('VMD-CNN-LSTM 集成预测结果与实际值比对', fontsize=20)
    plt.grid(True)
    plt.xlabel('时间步 (测试集)', fontsize=18)
    plt.ylabel('电力物资需求量(kg)', fontsize=18)
    plt.legend(fontsize=16)
    # 保存图像
    plt.savefig(os.path.join(TARGET_DIR, 'VMD-CNN-LSTM_集成预测结果与实际值比对.png'), dpi=300, bbox_inches='tight')
    plt.show()
