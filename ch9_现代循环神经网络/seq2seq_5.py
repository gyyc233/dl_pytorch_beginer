import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

import sys
sys.path.append('./ch9_现代循环神经网络') # based on dl_pytorch_beginer/

import encoder_decoder_4 as encoder_decoder_4


# 使用了嵌入层（embedding layer） 来获得输入序列中每个词元的特征向量
# 嵌入层的权重是一个矩阵， 其行数等于输入词表的大小（vocab_size）， 其列数等于特征向量的维度（embed_size）

#@save
class Seq2SeqEncoder(encoder_decoder_4.Encoder):
    """用于序列到序列学习的循环神经网络编码器,采用了多层门控循环单元"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        '''
        vocab_size: 输入词汇表的大小
        embed_size: 词嵌入向量的维度
        num_hiddens: RNN隐藏单元数量
        num_layers: RNN的层数
        dropout: 丢弃概率
        '''
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层：词汇索引转换为密集的向量表示，权重形状 [vocab_size, embed_size]
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 使用GRU的RNN
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        # permute 维度重排
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


# 使用一个两层门控循环单元编码器，其隐藏单元数为16，给定一小批量的输入序列X(批量大小4，时间步7)
# 在完成所有时间步后， 最后一层的隐状态的输出是一个张量，形状是[时间步数，批量大小，隐藏单元数]
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval() # 编码器设置为评估模式，会关闭 dropout
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
print('output.shape: ', output.shape)


# 使用另一种RNN作为解码器，在输出序列上的任意时间步t'，将来自上一时间步的输出与上下文变量作为输入
# 然后再当前时间步将它们与上一隐状态转换为新的隐状态
# 在获得解码器的隐状态之后， 我们可以使用输出层和softmax操作

class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size) # 嵌入层，将词元转换为密集向量表示
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout) # 多层门控循环单元
        self.dense = nn.Linear(num_hiddens, vocab_size) # 全连接层，将隐状态映射到词汇表大小的输出空间

    def init_state(self, enc_outputs, *args):
        # 状态初始化：接收编码器的输出，返回编码器最后的隐状态作为解码器的输入
        # 实现了编码器与解码器之间的状态传递
        return enc_outputs[1]

    def forward(self, X, state):
        #  permute(1, 0, 2) 调整维度顺序，从 (batch_size, num_steps, embed_size) 转换为 (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2) # 通过嵌入层处理并调整维度顺序以适应RNN的的输入
        # state[-1] 获取编码器最后层的最后隐藏状态作为上下文
        # repeat 将上下文向量复制 X.shape[0] 次，使其在时间步维度上与X匹配
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 在特征维度（第2维）上将嵌入向量 X 和上下文向量 context 拼接，拼接后的维度为 (num_steps, batch_size, embed_size + num_hiddens)
        X_and_context = torch.cat((X, context), 2)

        # 将拼接后的输入和当前状态传入 GRU 网络
        output, state = self.rnn(X_and_context, state)

        # 将隐藏状态映射到词汇表大小的输出空间
        # 使用 permute(1, 0, 2) 调整维度顺序回 (batch_size, num_steps, vocab_size)，便于后续处理
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
print('output.shape: ', output.shape) 
print('output.shape: ', state.shape)


# 损失函数
#@save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项
    valid_len: 有效长度张量，指定每个样本的有效序列长度
    value: 用于填充屏蔽位置的值,默认为0
    """
    maxlen = X.size(1) # 时间步长度
    print('maxlen: ', maxlen)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value # 取反掩码
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
seq_mask = sequence_mask(X, torch.tensor([1, 2]))
# valid_len=[1, 2] 表示第一行只有1个有效元素，第二行有2个有效元素
# 结果会将第一行的第2、3个元素和第二行的第3个元素屏蔽（设为0）
print('seq_mask: ', seq_mask)
