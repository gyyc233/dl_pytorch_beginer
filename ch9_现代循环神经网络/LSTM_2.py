import torch
from torch import nn
from d2l import torch as d2l

import sys
# 将脚本目录添加到系统路径中
sys.path.append('./ch8_循环神经网络_rnn') # based on dl_pytorch_beginer/

import rnn_4 as rnn_4
import rnn_config as rnn_config
import rnn_concise_5 as rnn_concise_5

# 长短记忆网络 LSTM，引入了记忆元(memory cell), 为控制记忆元设计了三种门，输出门，输入门，遗忘门
# 当前时间步的输入和前一个时间步的隐状态 作为数据送入LSTM
# 它们由三个具有sigmoid激活函数的全连接层处理， 以计算输入门、遗忘门和输出门的值
# 记忆元与候选记忆元
# 使用输出门与记忆元更新隐状态

def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = rnn_config.SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

batch_size, num_steps = 32, 35 # 每个批次的样本数量与每个序列的时间长度
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        # 返回指定形状的的正态分布随机数并*0.01,返回3个装量组成的元组
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# LSTM 的隐状态需要返回一个额外的记忆元， 单元的值为0，形状为（批量大小，隐藏单元数）
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


# 提供三个门和一个额外的记忆元，但是只有隐状态才会传到输出层，而记忆元不直接参与输出计算
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i) # 输入门
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f) # 遗忘门
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o) # 输出门
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c) # 候选记忆元
        C = F * C + I * C_tilda # 使用输入门、遗忘门、候选记忆元，更新记忆元
        H = O * torch.tanh(C) # 用输出门和新的记忆元更新隐状态
        Y = (H @ W_hq) + b_q # 用隐状态更新输出
        outputs.append(Y)

    # 所有时间步输出的拼接结果，并输出最终的隐状态和记忆元
    return torch.cat(outputs, dim=0), (H, C)


# 
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = rnn_concise_5.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
rnn_4.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


# 简洁实现
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = rnn_concise_5.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
rnn_4.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
