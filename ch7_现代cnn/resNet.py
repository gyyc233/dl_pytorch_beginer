# ResNet 残差网络：每个附加层都应该更容易地包含原始函数作为其元素之一
# 残差块将函数f变为包含一个简单的线性项和一个复杂的非线性项的相加
# DenseNet 稠密链接网络则是使用链接而不是相加
#   在跨层连接上，不同于ResNet中将输入与输出相加，稠密连接网络（DenseNet）在通道维上连结输入与输出

# 残差块 沿用了VGG完整的3*3 卷积层设计
# 残差块里首先有2个有相同输出通道数的3*3卷积，每个卷积后面接上一个批量归一化与relu激活函数, 然后我们通过跨层数据通路，跳过这2个卷积运算，将输入直接加在最后的ReLU激活函数前(使用1*1卷积对输入的深度进行升维或者降维)

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 残差块
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            # 1*1卷积核
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # 卷积1->BN1-ReLU1-卷积2-BN2
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        # 添加残差块
        if self.conv3:
            X = self.conv3(X)
        Y += X

        # 补上ReLU2
        return F.relu(Y)


blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape

# 增加输出通道数的同时，减半输出的高和宽
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape

# ResNet模型

# 对第一个模块做处理,第一个模块的通道数同输入通道数一致
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    '''
    n个残差块
    '''
    blk = []
    # 一个block包含两个残差块，首个残差块要做尺寸减半，通道翻倍
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 做长宽减半、通道数翻倍
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# * 表示解包
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)) # first_block=True 表示第一个模块的通道数和输入通道数一致
# 后面都是尺寸减半，通道翻倍
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# 最后在ResNet中加入全局平均汇聚层，以及全连接层输出
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

# 观察一下ResNet中不同模块的输入形状是如何变化的
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)


lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
