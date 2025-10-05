import torch
from torch import nn
from d2l import torch as d2l
import a_3_softmax_regression_scratch

# softmax回归的简洁实现
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 只需在Sequential中添加一个输入784，输出10的全连接层，仍然以均值0和标准差0.01随机初始化权重
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状 flatten层将输入“展平”成二维输入[batch_size, 28 * 28]
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

# 在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数
loss = nn.CrossEntropyLoss(reduction='none')

# train
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
a_3_softmax_regression_scratch.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)