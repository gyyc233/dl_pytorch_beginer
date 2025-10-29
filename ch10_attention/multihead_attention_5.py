import math
import torch
from torch import nn
from d2l import torch as d2l

import sys
sys.path.append('./ch10_attention')
import attention_scoring_function_3 as attention_scoring_function_3

# 多头注意力
# 使用独立学习得到的多组不同的线性投影来变换查询Q，键K和值V，然后，这组变换后的查询、键和值将并行地送到注意力汇聚中
# 最后，将h个注意力汇聚的输出拼接在一起， 并且通过另一个可以学习的线性投影进行变换,以产生最后的输出

# 对于h个注意力汇聚输出，每种注意力汇聚都被称为一个头


# 为了能够使多个头并行计算，定义了以下两个转置函数，transpose_output函数反转了transpose_qkv函数的操作

#@save
def transpose_qkv(X, num_heads):
    """
    为了多注意力头的并行计算而变换形状
    通过重新排列张量的维度来让多个注意力头可以同时进行计算
    X shape:[batch_size, seq_len, num_hiddens]
    """
    # first reshape: [batch_size, seq_length, num_heads, num_hiddens/num_heads]
    # 将最后一个维度（num_hiddens）分割成 num_heads 和 num_hiddens/num_heads 两部分
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 轴交换 [batch_size, num_heads, seq_length, num_hiddens/num_heads]
    X = X.permute(0, 2, 1, 3)

    # second reshape: [batch_size*num_heads, seq_length, num_hiddens/num_heads]
    # 将批次与头数合并为一个维度
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    # [batch_size*num_heads, seq_length, num_hiddens/num_heads] --> [batch_size, num_heads, seq_length, num_hiddens/num_heads]
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # [batch_size, seq_length, num_heads, num_hiddens/num_heads]
    X = X.permute(0, 2, 1, 3)
    # [batch_size, seq_length, num_hiddens]
    return X.reshape(X.shape[0], X.shape[1], -1)


#@save
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        '''
        key_size: 键的特征维度
        query_size: 查询的特征维度
        value_size: 值的特征维度
        num_hiddens: 隐藏层的维度
        num_heads: 多头注意力的个数
        dropout: 丢弃概率
        bias: 是否使用偏置
        '''
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = attention_scoring_function_3.DotProductAttention(dropout) # 点击注意力汇聚
        # Q K V 的线性变换层
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        # 输出的线性变换层
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)

        # 每个输入都经过相应线性层进行变换
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # 注意力计算
        output = self.attention(queries, keys, values, valid_lens)

        # 多头输出合并
        output_concat = transpose_output(output, self.num_heads)

        # 输出前应用输出线性层做变换
        return self.W_o(output_concat)

# 基于多头注意力对一个张量完成自注意力的计算
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()


# 
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
attention(X, Y, Y, valid_lens).shape
