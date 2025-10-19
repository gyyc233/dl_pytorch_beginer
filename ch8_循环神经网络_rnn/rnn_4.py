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
import re
import collections
import random
from d2l import torch as d2l
import rnn_config

# RNN的从零开始实现
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = rnn_config.SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

batch_size, num_steps = 32, 35 # 每个批次的样本数量与每个序列的时间长度
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

print('vocab: ', vocab)
print('vocab len: ', len(vocab))

# 将词元编码为one-hot
F.one_hot(torch.tensor([0,2]),len(vocab))
X=torch.arange(10).reshape(2,5) # 数据形状：(批量大小,时间步数)
F.one_hot(X.T, 28).shape # (5, 2, 28)，这样就把时间维度5放到前面 (时间，批量大小，样本特征长度)

# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
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


# 定义RNN网络模型,在初始化时返回隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    '''
    初始化隐藏层状态
    '''
    # 在初始化时返回隐状态, 返回是一个张量，张量全用0填充， 形状为（批量大小，隐藏单元数）
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# 定义RNN模型,描述如何在一个时间步内计算隐状态与输出
def rnn(inputs, state, params):
    '''
    inputs: (时间步数量，批量大小，词表大小)
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


# 使用一个类包装并存储RNN的参数
class RNNModelScratch: #@save
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens =vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self,X,state):
        '''
        call是一个特殊方法，它的作用是让类的实例可以像函数一样被调用
        X: (p批量大小，时间步数)
        '''
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32) # 变成浮点数
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

# 一个样例，观察输出是否符合要求
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print('Y.shape: ',Y.shape,', len(new_state): ', len(new_state), 'new_state[0].shape: ', new_state[0].shape)


def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix(字符串)后面生成新字符
    prefix: 起始字符串，用于引导生成过程
    num_preds: 要预测的字符数量
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期，没有输出，模型中的隐状态会更新
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # 预热期结束后，隐状态的值通常会比刚开始的初始值要更适合预测
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 我们还没有训练网络，它会生成荒谬的预测结果
predict = predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
print(predict)


def grad_clipping(net, theta):  #@save
    """裁剪梯度，防止梯度爆炸问题
    theta: 限制梯度的L2范数
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    # 计算所有参数梯度的L2范数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    # 当梯度范数超过阈值 theta 时才进行裁剪
    if norm > theta:
        for param in params:
            # 梯度乘以缩放因子 theta / norm，使得裁剪后不超过theta
            param.grad[:] *= theta / norm


# 训练
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 累加训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 在后续迭代中分离隐状态
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    # 通过detach_()防止梯度在时间步之间无限回传
                    s.detach_()

        # 前向传播获取预测结果
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)

        # 计算损失
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward() # 反向传播
            grad_clipping(net, 1) # 应用梯度裁剪
            # 因为已经调用了mean函数
            updater(batch_size=1)

        # 累计指标
        metric.add(l * y.numel(), y.numel())
    
    # 返回困惑度与训练速度
    return math.exp(metric[0] / metric  [1]), metric[1] / timer.stop()


#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())


# 使用随机抽样方法
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
