import pandas as pd
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
true_coef2 = 0
true_coef3 = 7
true_intercept = 7
polynomial = 4

# 因变量Y（目标），根据真实参数和自变量X计算得到，并添加一些随机噪声
Y = true_coef1 * X**3 + true_coef2 * X**2 + true_coef3 * X + true_intercept + np.random.randn(n_samples, 1)

fig, axs = plt.subplots(1, 1, figsize=(14, 5))
axs.scatter(X, Y)

turns = 100000
batches = 10000
alpha = 0.01
alpha_copy = alpha

def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch

def J(y_hat: np.array, y: np.array): 
    m = y.shape[0]
    return 1 / (2 * m) * np.sum((y_hat - y.reshape(1, m))**2)

def Hat(p: np.array, x: np.array): 
    m = x.shape[0]
    return p.reshape(1, polynomial).dot(np.vstack([(x**3).reshape(1, m), (x**2).reshape(1, m), x.reshape(1, m), np.ones((1, m))]))

def delta(p: np.array, x: np.array, y: np.array):
    m = y.shape[0]
    return 1 / m * (Hat(p, x) - y.reshape(1, m)).dot(np.hstack([x**3, x**2, x, np.ones((m, 1))]))

# 无正则化
p = np.array([1, 1, 1, 1])  # w1, w2, w3, b
trace = []
for i in tm.trange(turns):
    alpha = alpha_copy / (np.log(i + np.e))
    residual = J(Hat(p, X), Y)
    p = p - alpha * delta(p, X, Y)
    trace.append([i, p, residual])

for i in batched(trace, batches):
    print(*i[-1])

p = p.squeeze()
w1, w2, w3, b = p[0], p[1], p[2], p[3]

axs.plot([i for i in np.linspace(0, rng, rng*40)], 
         [w1 * i**3 + w2 * i**2 + w3 * i + b for i in np.linspace(0, rng, rng*40)], color="red")

# L1 Regression
lmd =np.array([10,10,0,0])

def J_with_l1(p: np.array, x: np.array, y: np.array):
    m = y.shape[0]
    return 1 / (2 * m) * np.sum((Hat(p, x) - y.reshape(1, m))**2) + 1* np.sum(np.abs(p))

def delta_with_l1(p: np.array, x: np.array, y: np.array):
    m = y.shape[0]
    gradient = 1 / m * (Hat(p, x) - y.reshape(1, m)).dot(np.hstack([x**3, x**2, x, np.ones((m, 1))]))
    regularization = lmd * np.sign(p)
    return gradient + regularization

p = np.array([1, 1, 1, 1])  # w1, w2, w3, bS
trace = []

for i in tm.trange(turns):
    alpha = alpha_copy / (np.log(i + np.e))
    residual = J_with_l1(p, X, Y)
    p = p - alpha * delta_with_l1(p, X, Y)
    trace.append([i, p, residual])

for i in batched(trace, batches):
    print(*i[-1])

p = p.squeeze()
w1, w2, w3, b = p[0], p[1], p[2], p[3]

axs.plot([i for i in np.linspace(0, rng, rng*40)], 
         [w1 * i**3 + w2 * i**2 + w3 * i + b for i in np.linspace(0, rng, rng*40)], color="green")

plt.show()
