import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(1337)


# 生成数据
X = np.linspace(-1, 1, 200)  # 在返回（-1, 1）范围内的等差序列
np.random.shuffle(X)  # 打乱顺序
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))  # 生成Y并添加噪声
# plot
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]  # 前160组数据为训练数据集
X_test, Y_test = X[160:], Y[160:]  # 后40组数据为测试数据集

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 选定loss函数和优化器
# model.compile(loss='mse', optimizer='sgd')


# Adam (short for Adaptive Moment Estimation)
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.01))

# 训练过程
print('Training -----------')
history = model.fit(X_train, Y_train, epochs=100, verbose=1)

# 画出损失函数的图
plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnidute")
plt.plot(history.history['loss'])
plt.show()

# 测试过程
print('Testing ------------')
lost = model.evaluate(X_test, Y_test, batch_size=40, verbose=0)

print('test cost:', lost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# 将训练结果绘出
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
