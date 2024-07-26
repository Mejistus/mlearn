import numpy as np
import matplotlib.pyplot as plt
import tqdm as tm

# 设置随机种子以确保结果可复现
np.random.seed(0)

# 样本数量
n_samples = 100

# 自变量X（特征），这里使用线性递增的序列并添加随机噪声
X = np.linspace(0, 2, n_samples) + np.random.randn(n_samples) * 0.001
X = X.reshape(-1, 1)  # 确保X是二维数组

# 真实参数
true_coef = 2.5
true_intercept = 1.0

# 因变量Y（目标），根据真实参数和自变量X计算得到，并添加一些随机噪声
Y = true_coef * X.squeeze() + true_intercept + np.random.randn(n_samples) * 0.5
Y = Y.reshape(-1, 1)  # 确保Y也是二维数组，以便后续计算

# 梯度下降参数
a = 1
learning_rate = a
iterations = 100000

# 初始化权重和偏置
w = 0
b = 0

# 梯度下降算法


def compute_gradients(X, Y, w, b):
    m = len(X)
    y_pred = w * X + b
    dw = 1/m * np.dot(X.T, (y_pred - Y))
    db = 1/m * np.sum(y_pred - Y)
    return dw, db


def gradient_descent(X, Y, w, b, learning_rate, iterations):
    trace = []
    for i in tm.trange(iterations):
        dw, db = compute_gradients(X, Y, w, b)
        learning_rate = a / (1+np.log(i+1))
        w -= learning_rate * dw
        b -= learning_rate * db
        residual = 1/(2*len(X)) * np.sum((w*X + b - Y)**2)
        trace.append([i, w, b, residual])
    return w, b, trace



w, b, trace = gradient_descent(X, Y, w, b, learning_rate, iterations)

for i, w, b, residual in trace[::1000]:
    print(i, w, b, residual)

plt.scatter(X, Y)
x_line = np.linspace(0, 2, 100)
final_y_line = np.squeeze(w * x_line + b)
plt.plot(x_line, final_y_line, 'r-', label='Final line')
plt.legend()
plt.show()
