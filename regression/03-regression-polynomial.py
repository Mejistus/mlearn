# !pip install pandas matplotlib numpy tqdm
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
X = 2 * np.random.rand(n_samples,1)  # 生成0到2之间的随机数
# 真实参数
true_coef1 = 20
true_coef2 = 7
true_intercept = 10.0
# 因变量Y（目标），根据真实参数和自变量X计算得到，并添加一些随机噪声
# + np.random.randn(n_samples, 1) * 0.5
Y = true_coef1 * X**2 +true_coef2 * X+ true_intercept + np.random.randn( n_samples,1)
# X, Y = X.squeeze(), Y.squeeze()
# print(np.shape(X), np.shape(Y))
def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch
def J(p:np.array,x:np.array, y:np.array): 
    m=y.shape[0]
    return 1/(2*m)*np.sum((Hat(p,x)-y.reshape(1,m))**2)
def Hat(p:np.array,x:np.array): 
    m=x.shape[0]
    return p.reshape(1,3).dot(np.vstack([(x**2).reshape(1,m),x.reshape(1,m),np.ones((1,m))]))
def delta(p:np.array,x:np.array,y:np.array):
    m=y.shape[0]
    return 1/m*(Hat(p,x)-y.reshape(1,m)).dot(np.hstack([x**2,x,np.ones((m,1))]))
p=np.array([1,1,1])# w1,w2,b
alpha=1
alpha_copy=alpha
trace = []
turns = 10000
batches = 1000
for i in tm.trange(turns):
    alpha=alpha_copy/(np.log(i+np.e))
    residual = J(p,X,Y)
    p=p-alpha*delta(p,X,Y)
    trace.append([i, p, residual])

for i in batched(trace, batches):
    print(*i[-1])

flag = 1
p=p.squeeze()
w1,w2,b=p[0],p[1],p[2]
if flag:
    plt.scatter(X, Y)
    plt.plot([i for i in np.linspace(0, 2, 100)],
            [w1*i**2+w2*i+b for i in np.linspace(0, 2, 100)],color="red")
    plt.show()
