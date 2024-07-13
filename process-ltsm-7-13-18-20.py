import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import talib

# 假设数据已经加载到一个名为df的Pandas DataFrame中，并且按时间顺序排序
df = pd.read_csv('CFFEX.IC_1min_copy.csv')

# 数据预处理
data = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_oi', 'close_oi']]

# 异常值处理：用每一列的平均值填充异常值
for column in ['open', 'high', 'low', 'close', 'volume', 'open_oi', 'close_oi']:
    mean_value = data[column].mean()
    data[column] = data[column].replace([np.inf, -np.inf], np.nan)
    data[column].fillna(mean_value, inplace=True)

# 预测的时间间隔
future_steps = 30

# 特征缩放
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume', 'open_oi', 'close_oi']])

# 准备训练数据和标签
X_train = []
y_train = []
for i in range(len(data) - future_steps):
    X_train.append(scaled_data[i:i + future_steps])
    y_train.append(scaled_data[i + future_steps])

X_train, y_train = np.array(X_train), np.array(y_train)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(future_steps, X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=X_train.shape[2]))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 创建一个 ModelCheckpoint 回调函数，保存训练好的模型
checkpoint = ModelCheckpoint('best_lstm_model.weights.h5', monitor='loss', save_best_only=True, save_weights_only=True, mode='min')

# 将数据划分为训练集和验证集
split = int(0.8 * len(X_train))
X_train_data, X_val_data = X_train[:split], X_train[split:]
y_train_data, y_val_data = y_train[:split], y_train[split:]

# 训练模型
history = model.fit(X_train_data, y_train_data, validation_data=(X_val_data, y_val_data), epochs=50, batch_size=32, callbacks=[checkpoint])

# 评估模型
train_pred = model.predict(X_train_data)
val_pred = model.predict(X_val_data)

train_mse = mean_squared_error(y_train_data, train_pred)
train_mae = mean_absolute_error(y_train_data, train_pred)
val_mse = mean_squared_error(y_val_data, val_pred)
val_mae = mean_absolute_error(y_val_data, val_pred)

print(f'Train MSE: {train_mse}, Train MAE: {train_mae}')
print(f'Validation MSE: {val_mse}, Validation MAE: {val_mae}')

# 使用训练好的模型进行预测
new_model = Sequential()
new_model.add(LSTM(units=50, return_sequences=True, input_shape=(future_steps, X_train.shape[2])))
new_model.add(LSTM(units=50))
new_model.add(Dense(units=X_train.shape[2]))

new_model.compile(optimizer='adam', loss='mean_squared_error')

# 载入最好的模型
new_model.load_weights('best_lstm_model.weights.h5')

# 初始化预测结果列表
predicted_data = []

# 使用最近的历史数据进行多步预测
current_data = scaled_data[-future_steps:].reshape((1, future_steps, scaled_data.shape[1]))

for _ in range(future_steps):
    prediction = new_model.predict(current_data)
    predicted_data.append(prediction[0])
    current_data = np.append(current_data[:, 1:, :], prediction.reshape((1, 1, X_train.shape[2])), axis=1)

predicted_data = np.array(predicted_data)

# 反向转换预测结果
predicted_data = scaler.inverse_transform(predicted_data.reshape(-1, X_train.shape[2]))
predicted_prices = predicted_data[:, 3]  # 假设只关注收盘价

# 定义一个函数计算MACD并判断是否有交点
def check_macd_cross(prices):
    macd, signal, _ = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
    if len(macd) < 2 or len(signal) < 2:
        return False
    # 检查最后两个点是否有交点
    if (macd[-2] < signal[-2] and macd[-1] > signal[-1]) or (macd[-2] > signal[-2] and macd[-1] < signal[-1]):
        return True
    return False

# 确保输入数组的类型是 float64
predicted_prices = predicted_prices.astype(np.float64)

# 使用函数检查预测的价格是否有MACD交点
has_cross = check_macd_cross(predicted_prices)
print("MACD Cross:", has_cross)
