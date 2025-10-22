import torch
from torch import nn
from d2l import torch as d2l

import sys
sys.path.append('./ch8_循环神经网络_rnn') # based on dl_pytorch_beginer/

import rnn_4 as rnn_4
import rnn_config as rnn_config
import rnn_concise_5 as rnn_concise_5

# 深度循环神经网络 每个隐状态都连续地传递到当前层的下一个时间步和下一层的当前时间步
# 深度循环神经网络需要大量的调参（如学习率和修剪） 来确保合适的收敛，模型的初始化也需要谨慎

def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = rnn_config.SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

batch_size, num_steps = 32, 35 # 每个批次的样本数量与每个序列的时间长度
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2 # 隐藏层数为2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = rnn_concise_5.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
rnn_4.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)
