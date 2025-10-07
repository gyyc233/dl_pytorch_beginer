import torch
from torch import nn

# bash nvidia-smi

if torch.cuda.is_available():
    print('usage GPU')
    print('usage gpu: ',torch.cuda.get_device_name(0))
else:
    # 好像默认装的是CPU版本了，要换成GPU
    print('usage CPU')

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()

# 1. 张量与所在设备
x = torch.tensor([1, 2, 3])
x.device

# 存储在GPU上
X = torch.ones(2, 3, device=try_gpu())
X

Y = torch.rand(2, 3, device=try_gpu())
Y

Z = X.cuda(0)
print(X)
print(Z)

# 2. 模型与所在设备 神经网络模型可以指定设备
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
net(X) # 网络与参数都在GPU上
net[0].weight.data.device
