import torch
from torch import nn
from d2l import torch as d2l

# LeNet 模型demo
# 在卷积神经网络中，我们组合使用卷积层、非线性激活函数和汇聚层
# 为了构造高性能的卷积神经网络，我们通常对卷积层进行排列，逐渐降低其表示的空间分辨率，同时增加通道数
# 卷积块编码得到的表征在输出之前需由一个或多个全连接层进行处理


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), # 图像深度1变成6
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), # 输入通道数由6变成16
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(), # 展平为[batch_size, 16 * 5 * 5]二维
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(), # 三次全连接，最后归类到10类
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))


X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# 逐层打印模型信息
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)

# train
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


# evaluate 验证评估
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) # 均匀分布初始化
    net.apply(init_weights) # 对网络所有module应用init_weights
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    # 每个epoch会遍历整个训练数据集一次
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train() # 将模型设置为训练模式

        for i, (X, y) in enumerate(train_iter): # 遍历训练数据加载器中的每个批次
            timer.start() 
            optimizer.zero_grad() # 如果没有optimizer.zero_grad()，梯度会在各个批次batch间累积
            X, y = X.to(device), y.to(device)
            y_hat = net(X) # 前向传播，得到预测结果
            l = loss(y_hat, y) # 计算交叉熵损失
            l.backward() # 反向传播，更新模型参数的梯度
            optimizer.step() # 参数更新

            # 在无梯度计算的上下文中累加
            with torch.no_grad():
                # 累加损失、准确率和样本数量
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()

            # 计算训练损失、训练准确率和测试准确率
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
