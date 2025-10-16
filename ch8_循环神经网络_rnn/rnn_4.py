# import torch
# from d2l import torch as d2l

# X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
# H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))

# T1 = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
# T2 = torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))

# print('T1: ',T1)
# print('T2: ',T2)

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# RNN的从零开始实现

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

print('vocab len: ', len(vocab))

# 将词元编码为one-hot
F.one_hot(torch.tensor([0,2]),len(vocab))
X=torch.arange(10).reshape(2,5)
F.one_hot(X.T, 28).shape # (5, 2, 28)

# 初始化模型参数
def get_params(vocab_size  , num_hiddens, device):
    '''
    vocab_size : 词表大小
    num_hiddens : 隐藏单元数，是可调的超参
    '''
    num_inputs = num_outputs = vocab_size # 输入维度和输出维度都等于词汇表大小

    def normal(shape):
        '''
        生成指定形状的随机权重矩阵
        '''
        return torch.randn(size=shape, device=device) * 0.01
    
    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens)) # 创建输入层到隐藏层的权重矩阵
    W_hh = normal((num_hiddens,num_hiddens)) # 隐藏层到隐藏层的权重矩阵（时间递归链接）
    b_h = torch.zeros(num_hiddens, device=device) # 隐藏层的偏置向量

    # 输出层参数
    W_hq = normal((num_hiddens,num_outputs)) # 隐藏层到输出层的权重矩阵
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True) # 启用梯度计算
    
    return params


# 定义RNN网络模型
def init_rnn_state(batch_size, num_hiddens, device):
    '''
    初始化隐藏层状态
    '''
    # 在初始化时返回隐状态, 返回是一个张量，张量全用0填充， 形状为（批量大小，隐藏单元数）
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# 定义RNN模型,描述如何在一个时间步内计算隐状态与输出
def rnn(intputs, state, params):
    '''
    inputs的形状：(时间步数量，批量大小，词表大小)
    state: 初始隐藏状态
    params: 模型参数列表，包括权重和偏置
    '''
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state # 获取当前隐藏状态
    outputs = []

    # X的形状：(批量大小，词表大小)
    for X in inputs: # 对输入序列中的每个时间步 X 进行处理
        # 计算隐藏层输出
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h) # 激活函数使用tanh
        # 计算输出层输出
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)

    # 将所有时间步的输出沿时间步维拼接，并返回隐藏状态（元组）
    return torch.cat(outputs, dim=0), (H,)
