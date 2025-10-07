import torch
from torch import nn
from d2l import torch as d2l


# 汇聚层 or 池化层
# 1. 与卷积层类似，汇聚层运算符由一个固定形状的窗口组成，该窗口根据其步幅大小在输入的所有区域上滑动，为固定形状窗口
# 2. 不同于卷积层中的输入与卷积核之间的互相关计算，汇聚层不包含参数
# 3. 池运算是确定性的，我们通常计算汇聚窗口中所有元素的最大值或平均值。这些操作分别称为最大汇聚层（maximum pooling）和平均汇聚层（average pooling）

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))

pool2d(X, (2, 2), 'avg') # 平均汇聚层

# 汇聚层也有padding和stride参数
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))

pool2d = nn.MaxPool2d(3)
pool2d(X)

# 默认情况下，(深度学习框架中的步幅与汇聚窗口的大小相同)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)

# 汇聚层的输出通道数与输入通道数相同
X = torch.cat((X, X + 1), 1) # 在通道维度上连结张量X与X+1
print(X.shape) # (1, 2, 4, 4)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
print(X.shape)
