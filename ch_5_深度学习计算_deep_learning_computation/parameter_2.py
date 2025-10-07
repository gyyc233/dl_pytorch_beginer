import torch
from torch import nn

# 访问、修改、共享网络中的参数

# 单隐藏层的MLP
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

# 1. 第2个网络层的参数
print(net[2].state_dict())
# OrderedDict([('weight', tensor([[ 0.0546, -0.2078, -0.1583, -0.3005, -0.3417, -0.2544,  0.1366, -0.0123]])), ('bias', tensor([0.1247]))])
# 这个全连接层包含两个参数，分别是该层的权重和偏置。 两者都存储为单精度浮点数（float32）。 注意，参数名称允许唯一标识每个参数

# 2. 进一步访问参数的值
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
# <class 'torch.nn.parameter.Parameter'>
# Parameter containing:
# tensor([0.1247], requires_grad=True)
# tensor([0.1247])

# 由于我们还没有调用反向传播，所以参数的梯度处于初始状态
print(net[2].weight.grad == None)

# 3. 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
# 访问第2个网络层的的偏置
net.state_dict()['2.bias'].data

# 4. 将多个块相互嵌套后访问参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X) # X 为 2 * 4
print(rgnet)

# 访问第一个主要的块中、第二个子块的第一层的偏置项
rgnet[0][1][0].bias.data


# 参数初始化

# 1. 使用内置初始化 将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0
def init_normal(m):
    '''m 为nn.module '''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal) # 对net中的所有层应用init_normal进行初始化
net[0].weight.data[0], net[0].bias.data[0]

# 将所有参数初始化为给定的常数1
def init_constant(m):
    '''m 为nn.module '''
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)

# 对某些块应用不同的初始化方法
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier) # 用Xavier初始化方法初始化第一个神经网络层
net[2].apply(init_42) # 将第三个神经网络层初始化为常量值42
print(net[0].weight.data[0])
print(net[2].weight.data)

# 2. 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]

# 直接设置参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]

# 3. 参数共享 or 参数绑定 有时我们希望在多个层间共享参数
shared = nn.Linear(8, 8) # 共享层的名字
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 第三个和第五个神经网络层的参数是绑定的
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
