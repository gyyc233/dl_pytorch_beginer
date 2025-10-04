import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn # nn是神经网络的缩写

# 线性回归的简洁实现

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器 is_train 是否打乱顺序"""
    # 使用 data.TensorDataset(*data_arrays) 将特征和标签打包成数据集
    dataset = data.TensorDataset(*data_arrays)
    # data.DataLoader 创建数据加载器 返回一个 PyTorch DataLoader 对象，可以用于迭代访问批量数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 1. 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 2. 读取数据集 
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 使用iter构造Python迭代器，并使用next从迭代器中获取第一项
next(iter(data_iter))

# 3. 定义模型
# Sequential类将多个层串联在一起 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推
# 在下面的例子中，我们的模型只包含一个层，因此实际上不需要Sequential。 但是由于以后几乎所有的模型都是多层的，在这里使用Sequential会让你熟悉“标准的流水线”
# Linear 全连接层在Linear类中定义,
# 将两个参数传递到nn.Linear中, 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1
net = nn.Sequential(nn.Linear(2, 1))

# 4. 初始化模型参数
# net[0]选择网络中的第一个图层, 使用weight.data和bias.data方法访问参数
# 还可以使用替换方法normal_和fill_来重写参数值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 5. 定义损失函数 均方误差使用的是MSELoss类 返回所有样本损失的平均值
loss = nn.MSELoss()

# 6. 定义优化算法
# 小批量随机梯度下降算法是一种优化神经网络的标准工具， PyTorch在optim模块中实现了该算法的许多变种
# 要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。 小批量随机梯度下降只需要设置lr值，这里设置为0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 7. 训练
# 在每个迭代周期里，我们将完整遍历一次数据集（train_data）， 不停地从中获取一个小批量的输入和相应的标签。 对于每一个小批量，我们会进行以下步骤:
# - 通过调用net(X)生成预测并计算损失l（前向传播）
# - 通过进行反向传播来计算梯度
# - 通过调用优化器来更新模型参数
# 为了更好的衡量训练效果，我们计算每个迭代周期后的损失，并打印它来监控训练过程

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l=loss(net(X), y)
        trainer.zero_grad() # 清空优化器中参数的梯度缓存
        l.backward() # 计算损失函数相对于模型参数的梯度
        trainer.step() # 更新模型参数

    # 每轮结束后计算整体损失
    l=loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 比较生成数据集的真实参数和通过有限数据训练获得的模型参数
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
