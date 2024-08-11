# 感知机原理

感知机（Perceptron）是一种基本的二分类模型。它通过对输入特征的线性组合来做出分类决策。

## 数学模型

感知机的输出由输入向量的线性组合决定，公式如下：

$$
y = \text{sign}(w \cdot x + b)
$$

其中：

- $w$ 是权重向量
- $x$ 是输入向量
- $b$ 是偏置
- $\text{sign}(z)$ 是符号函数，定义为：

$$
\text{sign}(z) =
\begin{cases} 
+1 & \text{if } z \geq 0 \\
-1 & \text{if } z < 0
\end{cases}
$$

## 学习规则

感知机的学习规则基于调整权重和偏置，公式如下：

$$
w = w + \eta y_i x_i
$$

$$
b = b + \eta y_i
$$

$$ 其中，
\eta是学习率,\\ 
 y_i 是样本的真实标签。
$$

## 感知机的局限性

感知机只能处理线性可分的数据，对于非线性可分的数据无法找到合适的分类超平面。

## Python实现
```python 
class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

```
