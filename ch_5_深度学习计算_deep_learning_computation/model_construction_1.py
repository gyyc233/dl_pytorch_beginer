import torch
from torch import nn
from torch.nn import functional as F

# 层与块


# 1. 构造一个具有256个单元和relu激活函数的全连接隐藏层，然后是一个具有10个隐藏单元且不带激活函数的全连接输出层

import torch
from torch import nn
from torch.nn import functional as F

# nn.Sequential定义了一种特殊的Module，表示一个块的类， 它维护了一个由Module组成的有序列表
# net(X)是net.__call__(X)的简写，这里的前向传播函数：将列表中的每个块连接在一起，将每个块的输出作为下一个块的输入
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)


# 自定义块
# 1. 将输入数据作为其前向传播函数的参数
# 2. 通过前向传播函数来生成输出
# 3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的(也可以自己写)
# 4. 存储和访问前向传播计算所需的参数
# 5. 根据需要初始化模型参数

# 以下块包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        #  计算带有激活函数的隐藏表示，并输出其未规范化的输出值
        return self.out(F.relu(self.hidden(X)))
    

net = MLP()
net(X)


# 顺序块 Sequential的设计是为了把其他模块串起来
# 构建我们自己的简化的MySequential， 我们只需要定义两个关键函数：
# 1. 一种将块逐个追加到列表中的函数
# 2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 变量_modules中。_module的类型是OrderedDict 有序字典
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        # 每个添加的块都按照它们被添加的顺序执行
        for block in self._modules.values():
            X = block(X)
        return X
    
# 使用我们的MySequential类重新实现多层感知机
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)


# 混合搭配各种组合块的方法


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
