import torch
from torch import nn
from torch.nn import functional as F

# 张量、网络参数的保存读取
x = torch.arange(4)
torch.save(x, 'x-file') # 保存张量到文件

x2 = torch.load('x-file') # 从文件中读取张量
x2

# 保存多个张量
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)

# 保存字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2


# 模型参数的保存与加载 这个框架只保留模型参数，不保留模型结构，在加载时需要重新定义模型结构
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 将模型的参数存储在一个叫做“mlp.params”的文件中
torch.save(net.state_dict(), 'mlp.params')

# 为了恢复模型，我们[实例化了原始多层感知机模型的一个备份
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

# 由于两个实例具有相同的模型参数，在输入相同的X时， 两个实例的计算结果应该相同
Y_clone = clone(X)
print(Y_clone == Y)
