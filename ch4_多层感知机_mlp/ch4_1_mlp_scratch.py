import torch
from torch import nn
from d2l import torch as d2l
import train_ch3

# mlp多层感知机的从零开始实现

batch_size = 256 # 批量大小256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# Fashion-MNIST中的每个图像由28*28=784灰度像素值组成。 所有图像共分为10个类别
# 忽略像素之间的空间结构， 我们可以将每个图像视为具有784个输入特征 和10个类的简单分类数据集
# 实现一个具有单隐藏层的多层感知机， 它包含256个隐藏单元
num_inputs, num_outputs, num_hiddens = 784, 10, 256

def relu(X):
    """ReLU激活函数"""
    a = torch.zeros_like(X)
    return torch.max(X, a)


# model
def net(X):
    X = X.reshape((-1, num_inputs)) # X变为batch_size * num_inputs 256*784
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)


# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

if __name__ == "__main__":
    # 随机初始化权重与偏置
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    # 训练
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
