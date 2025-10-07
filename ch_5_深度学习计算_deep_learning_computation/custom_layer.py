import torch
import torch.nn.functional as F
from torch import nn

# 自定义层
# 研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。 有时我们会遇到或要自己发明一个现在在深度学习框架中还不存在的层。 在这些情况下，必须构建自定义层

# 1. 不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

# 合并到其它模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()

# 2. 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        '''
        in_units 输入的维度
        units 输出的维度
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
linear.weight

# 使用自定义层直接执行前向传播计算
linear(torch.rand(2, 5))

# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
