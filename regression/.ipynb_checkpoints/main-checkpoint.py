import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import tqdm as tm
#产生数据
import numpy as np  
# 设置随机种子以确保结果可复现  
np.random.seed(0)  
# 样本数量  
n_samples = 100  
# 自变量X（特征），这里我们简单地使用了一个线性递增的序列，并添加了一些随机噪声  
X = 2 * np.random.rand(n_samples, 1)  # 生成0到2之间的随机数  
# 真实参数  
true_coef = 2.5  
true_intercept = 1.3  
# 因变量Y（目标），根据真实参数和自变量X计算得到，并添加一些随机噪声  
Y = true_coef * X.squeeze() + true_intercept + np.random.randn(n_samples, 1) * 0.5  
# 现在我们有了数据集 (X, Y)，其中X是自变量，Y是因变量  
# 你可以使用这些数据来训练你的回归模型  
# 打印前几行数据以查看  
# print("X:\n", X[:5],np.shape(X))  
# print("Y:\n", Y[:5],np.shape(Y))

J=lambda y_hat,y,m:1/2*m*sum((y_hat-y)**2)
hat=lambda w,x,b:w*x+b
dw=lambda x,y,w,b,m:1/m*sum((hat(w,x,b)-y)*x)
db=lambda x,y,w,b,m:1/m*sum((hat(w,x,b)-y))

a,w,b,m=0.001,1,1,1  

for i in tm.trange(100):
    residual=J(hat(w,X,b),Y,m)
    
    print(residual)