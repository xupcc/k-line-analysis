# k-line-analysis
使用LSTM模型进行股票价格预测和MACD交叉检测
该项目涉及使用LSTM（长短期记忆）神经网络来预测股票价格并检测MACD（移动平均收敛/发散）交叉。代码利用了Python库，如TensorFlow、Keras、Pandas、NumPy和TA-Lib，进行数据预处理、模型训练和评估。

先决条件
要运行此代码，您需要安装以下Python库：

numpy
pandas
scikit-learn
tensorflow
talib
您可以使用pip安装这些库：


pip install numpy pandas scikit-learn tensorflow talib
数据准备
加载数据：
从名为 CFFEX.IC_1min_copy.csv 的CSV文件加载数据。数据集应按时间顺序排序。

数据预处理：

选择相关列：datetime、open、high、low、close、volume、open_oi、close_oi。
处理异常值，将其替换为各列的平均值。
使用 MinMaxScaler 将特征缩放到0到1之间的范围。
准备训练数据：

使用30分钟的窗口（即 future_steps = 30）创建训练序列。
将数据划分为训练集和验证集（80%训练，20%验证）。
LSTM模型
LSTM模型架构包括：

两个LSTM层，每层50个单元。
一个Dense层，单元数量与输入特征数量相同。
使用Adam优化器和均方误差损失函数进行编译。
训练和评估
模型训练50个周期，批量大小为32。
使用 ModelCheckpoint 回调函数保存基于训练损失的最佳模型权重。
使用均方误差（MSE）和平均绝对误差（MAE）在训练集和验证集上评估模型。
预测和MACD交叉检测
加载最佳模型：
加载最佳模型权重进行预测。

预测未来价格：

使用最后30分钟的历史数据迭代预测未来30分钟的价格。
将预测数据缩放回原始尺度。
MACD交叉检测：

使用 TA-Lib 计算MACD和信号线。
检查预测价格中的交叉点。