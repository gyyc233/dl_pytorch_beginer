import torch
from torch import nn
from d2l import torch as d2l

import sys
# 将脚本目录添加到系统路径中
sys.path.append('./ch8_循环神经网络_rnn') # based on dl_pytorch_beginer/

import rnn_4 as rnn_4
import rnn_config as rnn_config
import rnn_concise_5 as rnn_concise_5

# 门控循环单元 GRU
# 门控循环单元与普通的循环神经网络之间的关键区别在于：前者支持隐状态的门控，这意味着模型有专门的机制来确定应该何时更新隐状态， 以及应该何时重置隐状态，这些机制是可学习的，并且能够解决了上面列出的问题

# 重置门 R_t 与候选隐状态 H_t^，重置门中所有接近0的项，候选隐状态形式上接近MLP，所有接近1的项，候选隐状态形式上接近经典的循环神经网络
# 更新门 Z_t 与新的隐状态 H_t， 更新门中所有接近0的项，新的隐状态形式上接近候选隐状态，所有接近1的项，新的隐状态形式上接近上一次的隐状态
# 以上过程类似kalman fliter

# 1. 重置门有助于捕获序列中的短期依赖关系
# 2. 更新门有助于捕获序列中的长期依赖关系


def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = rnn_config.SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

batch_size, num_steps = 32, 35 # 每个批次的样本数量与每个序列的时间长度
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):  #@save
    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device=device))
    
    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    for param in params:
        param.requires_grad_(True)

    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def gru(inputs, state, params):
    '''
    定义门控循环单元模型
    '''
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z) # 更新门
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r) # 重置门
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h) # 候选隐状态
        H = Z * H + (1 - Z) * H_tilda # 有点像 kalman fliter
        Y = H @ W_hq + b_q # 输出层
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = rnn_concise_5.RNNModel(gru_layer, len(vocab))
model = model.to(device)
rnn_4.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
