import random
import torch
from d2l import torch as d2l

# 线性神经网络的从零开始实现


# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声 w 线性参数 b 偏置项 num_examples 样本数"""

    # X是矩阵 n*2 (num_examples, len(w))：这指定了输出张量的形状
    # num_examples：要生成的样本数量
    # len(w)：每个样本的特征数量，由权重向量 w 的长度决定
    X = torch.normal(0, 1, (num_examples, len(w))) # 正态分布
    y = torch.matmul(X, w) + b # 标签
    y += torch.normal(0, 0.01, y.shape) # 为标签添加正态分布噪声
    return X, y.reshape((-1, 1))


# 读取数据集
def data_iter(batch_size, features, labels):
    """读取小批量数据集 batch_size 批量大小 features 数据集所有的特征 labels 数据集所有的标签"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 打乱index顺序

    # range(0, num_examples, batch_size) 从0开始，到 num_examples 结束（不包含），以 batch_size 为步长进行迭代
    for i in range(0, num_examples, batch_size):
        # 获取当前批次索引 从i到步长结束或者末尾
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
    
        # yield 关键字返回每个批次的数据

        # return：执行后函数立即结束，返回一个值
        # yield：执行后函数暂停，返回一个值，但保留当前状态，下次调用时从暂停处继续执行
        yield features[batch_indices], labels[batch_indices]


# 定义模型
def linreg(X, w, b):
    """线性回归模型 X 特征 w 权重 b 偏置项"""
    # 用一个向量加一个标量时，因为广播机制，标量会作用到向量的每个分量
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失函数 y_hat 预测值 y 真实值"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# SGD优化算法
# @save 这是一个注释标记，表示保存该函数以供后续使用
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降 params 模型参数 lr 学习率 batch_size 批量大小"""
    # 尽管线性回归有解析解，但这里使用SGD小批量随机梯度下降

    # with torch.no_grad() 在更新参数时不需要计算梯度，避免影响梯度值
    with torch.no_grad():
        for param in params:
            # lr：学习率控制更新步长
            # 反向梯度更新
            param -= lr * param.grad / batch_size
            param.grad.zero_() # 梯度清零为下一次迭代做准备


true_w = torch.tensor([2, -3.4])
true_b = 8.1


# 1. 生成数据集
features, labels = synthetic_data(true_w, true_b, 1000)

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1);
# d2l.plt.show()

# 读取数据集测试
# batch_size = 20
# for X,y in data_iter(batch_size, features, labels):
#     print(X,'\n',y)
#     break


# 2. 初始化模型参数,启用梯度计算
# 均值为0、标准差为0.01的正态分布中采样随机数来初始化权重， 并将偏置初始化为0 会导致所有神经元在训练开始时具有相同的输出和梯度，从而无法有效地学习
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
# 将权重初始化为0
b=torch.zeros(1, requires_grad=True)


# 在初始化参数之后，我们的任务是更新这些参数，直到这些参数足够拟合我们的数据
# 每次更新都需要计算损失函数关于模型参数的梯度

# 3. 训练
# 在每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测
# 计算完损失后，我们开始反向传播，存储每个参数的梯度
# 最后，我们调用优化算法sgd来更新模型参数

# hyperparameters 超参数 3个
lr = 0.03
batch_size = 10
num_epochs = 3 # 迭代轮数

# 模型与损失函数
net = linreg
loss = squared_loss


for epoch in range(num_epochs):
    # 在每个迭代周期（epoch）中，使用data_iter函数遍历整个数据集， 并将训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward() # 自动微分计算梯度
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    
    # evaluation 评估本次迭代周期的训练结果
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 比较真实参数和通过训练学到的参数来评估训练的成功程度
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
