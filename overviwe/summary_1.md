- [线性回归](#线性回归)
  - [softmax](#softmax)
- [MLP](#mlp)
- [CNN](#cnn)

# 线性回归

- 定义线性回归模型`torch.matmul(X, w) + b`
- 初始化模型参数
- 生成数据集`y=Xw+b+噪声 w 线性参数 b 偏置项`
- 定义损失函数 均方损失 `(y_hat - y.reshape(y_hat.shape)) ** 2 / 2` 预测-真实
- 优化算法 小批量随机梯度下降SGD(梯度的反方向),需要先对参数进行反向传播，再结合学习率
  - `param -= lr * param.grad / batch_size` 这里的梯度是求和之后的，需要除以batch_size
  - 更新参数时不需要计算梯度
  - 跟新后结束后需要进行梯度清零

在每个迭代周期执行以下步骤，每个迭代周期都会训练所有的训练数据
1. 按照batch_size从数据集中获取小批量数据
2. 使用网络训练，得到输出 `nat`
3. 根据输出与参考值计算损失函数 `loss`, 然后对损失求和，损失最后往往都定义为标量
4. 对求和后的损失进行反向传播，计算损失对于参数的梯度 `backward`
5. 使用小批量随机梯度下降`SGD`更新参数 `param -= lr * param.grad / batch_size`
6. 本次batch_size遍历结束后，对本次迭代周期的参数进行评估
   1. 在每个epoch结束后，使用整个数据集计算当前模型的平均损失，监控训练过程

## softmax

用于多分类问题,损失函数使用交叉熵

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
```

# MLP

多层感知机

- Fashion-MNIST中的每个图像由28*28=784灰度像素值组成。 所有图像共分为10个类别
- 忽略像素之间的空间结构， 我们可以将每个图像视为具有784个输入特征 和10个类的简单分类数据集
- 添加了隐藏的全连接层，网络越深，可能会遇到梯度消失和梯度爆炸的问题
- 正则化技术啊：权重衰减`weight_decay`、暂退层 `dropout`
  - 权重衰减：系数*权重的L2范数家在损失中
  - 暂退层：随机丢弃隐藏层中的部分节点，比较主流的正则化方法
- 激活函数：`sigmoid`、`ReLU(比较稳定，在更深网络中难以引起梯度消失)`、`tanh`
- 损失函数一般是交叉熵

1. 当使用pytorch优化器时`torch.optim.SGD、torch.optim.Adam`,使用损失均值求梯度`l.mean().backward()` `updater.step()`
2. 线性回归中使用自定义SGD，使用梯度求和求损失`l.sum().backward()`，在后面使用自定义优化器函数通过除以批次大小来处理归一化`param -= lr * param.grad / batch_size`

延后初始化

1. 定义了网络架构，但没有指定输入维度，这里应用了框架的延后初始化（defers initialization），即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小
2. 一旦我们知道输入维数是20，框架可以通过代入值20来识别第一层权重矩阵的形状。 识别出第一层的形状后，框架处理第二层，依此类推，直到所有形状都已知为止。 注意，在这种情况下，只有第一层需要延迟初始化，但是框架仍是按顺序初始化的

# CNN

图像卷积是图像与卷积核做二维互相关计算，并没有把图像平展

```python
def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# 卷积层 对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

- 填充 padding 与步幅 stride (n-k+p+s)/s
- 汇聚层（池化层）pooling: 汇聚层不包含参数, 池运算是确定性的, 会改变输入宽高
  - 通常计算汇聚窗口中所有元素的最大值或平均值。这些操作分别称为最大汇聚层（maximum pooling）和平均汇聚层（average pooling）
  - 但是不会改变通道数
- 1 * 1卷积层：卷积核大小为1*1，通常用于改变通道数，如从3通道变成10通道
