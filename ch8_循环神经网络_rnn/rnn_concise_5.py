import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import rnn_config
import rnn_4

# RNN 简洁实现
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = rnn_config.SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

batch_size, num_steps = 32, 35 # 批量大小和时间步数
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
print('vocab size:', len(vocab))

# 构造一个具有256个隐藏单元的单隐藏层的RNN
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# 使用张量来初始化隐状态，它的形状是（隐藏层数，批量大小，隐藏单元数）
state = torch.zeros([1,batch_size,num_hiddens])
print('state size:', state.shape)

# 通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出
# rnn_layer的“输出”（Y）不涉及输出层的计算： 它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入
X= torch.rand(size=(num_steps, batch_size, len(vocab))) # 输入形状：(时间步数，批量大小，词表大小)
Y, state_new = rnn_layer(X, state)
print('Y size:', Y.shape, ', new state size:', state_new.shape)


# 为一个完整的循环神经网络模型定义了一个RNNModel类
# rnn_layer只包含隐藏的循环层，我们还需要创建一个单独的输出层
#@save
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        '''
        前向传播方法，结束输入数据与隐藏状态
        '''
        # X的形状是 [num_steps, batch_size, vocab_size]
        X = F.one_hot(inputs.T.long(), self.vocab_size) # long() 确保数据类型为长整型
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state) # Y是每个时间步的隐藏状态输出 [num_steps, batch_size, hidden_size]
        # 重塑后的形状 [num_steps * batch_size, hidden_size]

        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        # output的形状 [num_steps * batch_size, vocab_size]
        return output, state

    def begin_state(self, device, batch_size=1):
        '''
        状态初始化函数，为不同类型RNN（GRU、LSTM等）创建合适的初始隐藏状态
        支持多层和双向RNN的复杂状态结构
        '''
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
        

# 在训练模型之前，让我们基于一个具有随机权重的模型进行预测
print('model not training, just predict...')
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
rnn_4.predict_ch8('time traveller', 10, net, vocab, device)

# 训练
num_epochs, lr = 500, 1
rnn_4.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
