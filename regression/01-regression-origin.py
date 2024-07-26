import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tm
from itertools import islice
# 产生数据
# 设置随机种子以确保结果可复现
np.random.seed(0)
# 样本数量
n_samples = 100
# 自变量X（特征），这里我们简单地使用了一个线性递增的序列，并添加了一些随机噪声
X = 2 * np.random.rand(1, n_samples)  # 生成0到2之间的随机数
# 真实参数
true_coef = 2.5
true_intercept = 1.0
# 因变量Y（目标），根据真实参数和自变量X计算得到，并添加一些随机噪声
# + np.random.randn(n_samples, 1) * 0.5
Y = true_coef * X.squeeze() + true_intercept + np.random.randn(1, n_samples)*0.5
X, Y = X.squeeze(), Y.squeeze()
print(np.shape(X), np.shape(Y))

# print(X[:5], Y, sep="\n")


def J(w, x, b, y, m): return 1/(2*m)*sum((hat(w, x, b)-y)**2)
def hat(w, x, b): return w*x+b
def dw(x, y, w, b, m): return 1/m*sum((hat(w, x, b)-y)*x)
def db(x, y, w, b, m): return 1/m*sum((hat(w, x, b)-y))


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


f, w, b, m = 1, 0, 0, n_samples
a = f
trace = []
turns = 10000
batches = 100
for i in tm.trange(turns):
    a = f/(np.log(i+np.e))
    residual = J(w, X, b, Y, m)
    w, b = w-a*dw(X, Y, w, b, m), b-a*db(X, Y, w, b, m)
    trace.append([i, w, b, a, residual])
for i in batched(trace, batches):
    print(*i[-1])


flag = 1
if flag:
    plt.scatter(X, Y)
    # for i, w, b,a, residual in trace:
    #     if i % 100 == 0:
    #         plt.plot([i for i in np.linspace(0, 2, turns//batches)],
    #                  [w*i+b for i in np.linspace(0, 2, turns//batches)])

    plt.plot([i for i in np.linspace(0, 2, 100)],
            [w*i+b for i in np.linspace(0, 2, 100)],color="red")
    plt.show()
