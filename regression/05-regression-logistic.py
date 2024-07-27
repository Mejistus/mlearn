import numpy as np
import matplotlib.pyplot as plt
import tqdm as tm
from itertools import islice

# 产生数据
np.random.seed(0)
n_samples = 100
rng = 2
X = rng * np.random.rand(n_samples, 1)  # 生成0到rng之间的随机数

# 真实参数
true_coef1 = 0
true_coef2 = -4
true_coef3 = 5
true_intercept = -1
polynomial = 3

# 计算线性组合
linear_combination = true_coef1 * X**3 + true_coef2 * X**2 + true_coef3 * X + true_intercept

# 定义数值限制范围
cliprange = 500

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -cliprange, cliprange)))

# 应用逻辑函数
probabilities = sigmoid(linear_combination)

# 生成二分类的因变量Y（目标），根据概率生成0或1
Y = (probabilities >= 0.5).astype(int).astype(float)

# 可视化数据
fig, axs = plt.subplots(1, 1, figsize=(14, 5))
axs.scatter(X, Y, c=Y, cmap='viridis', edgecolors='k')
axs.set_xlabel("X")
axs.set_ylabel("Y")
axs.set_title('Generated Data for Logistic Regression')
# plt.show()

turns = 1000000
batches = 1000
alpha = 0.1
alpha_copy = alpha

def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch

def J(y_hat: np.array, y: np.array):
    m = y.shape[0]
    return 1 / (2 * m) * np.sum(np.square(y_hat - y.reshape(1, m)))

def Jpxy(p: np.array, x: np.array, y: np.array):
    m = y.shape[0]
    return 1 / (2 * m) * np.sum(np.square(Hat(p, X) - y.reshape(1, m)))

def Hat(p: np.array, x: np.array):
    m = x.shape[0]
    return sigmoid(p.reshape(1, polynomial).dot(
        np.vstack([
            np.clip((x**2).reshape(1, m), -cliprange, cliprange),
            np.clip(x.reshape(1, m), -cliprange, cliprange),
            np.ones((1, m))]
        )
    ))

def delta(p: np.array, x: np.array, y: np.array):
    m = y.shape[0]
    X_ = np.vstack([
        np.clip((x**2).reshape(1, m), -cliprange, cliprange),
        np.clip(x.reshape(1, m), -cliprange, cliprange),
        np.ones((1, m))]
    )  # 2,100
    temp = Hat(p, x) - y.reshape(1, m)
    temp = temp * Hat(p, x) * (1 - Hat(p, x))
    temp = temp.dot(X_.T)
    return 1 / m * temp

p = np.array([1, 1, 1])  # w3, b
trace = []

for i in tm.trange(turns):
    # alpha = alpha_copy / (np.log(i + np.e))
    residual = J(Hat(p, X), Y)
    p = p - alpha * delta(p, X, Y)
    trace.append([i, p, residual])

for i in batched(trace, batches):
    print(*i[-1])

p = p.squeeze()
w2, w3, b = p[0], p[1], p[2]

axs.plot([i for i in np.linspace(0, rng, rng * 40)],
         [sigmoid(w2*i**2 + w3 * i + b) for i in np.linspace(0, rng, rng * 40)], color="red")

plt.show()

