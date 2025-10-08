import torch
from torch import nn
from d2l import torch as d2l

# VGG visual geometry group 使用可复用的卷积块构造网络
# 深层且窄的卷积比浅层且宽的卷积更有效

# 经典卷积神经网络的基本组成部分是下面的这个序列
# 1. 带填充以保持分辨率的卷积层
# 2. 非线性激活函数，如ReLU
# 3. 汇聚层，如最大池化层

# 一个VGG块与之类似，由n个卷积层组成，后面再加上用于空间下采样的最大汇聚层

# VGG块
def vgg_block(num_convs, in_channels, out_channels):
    '''
    num_convs: 卷积层的数量
    in_channels: 输入通道数
    out_channels: 输出通道数
    '''
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels # 从下次开始输入通道与输出通道相等
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2)) # 加入汇聚层
    return nn.Sequential(*layers)


# 原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层
# 1+1+2+2+2 =8
# 第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512

# 卷积层个数与输出通道数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    '''
    VGG-11
    '''
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    # VGG块结束后再加上三个全连接层
    return nn.Sequential(
        # VGG的输出进行展平
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)


# 观察每层输出的相撞
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

# 模型训练 
ratio = 4 # 由于VGG-11比AlexNet计算量更大，因此我们构建了一个通道数较少的网络
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)


lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
