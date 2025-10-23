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


# 通过扩展softmax交叉熵损失函数来遮蔽不相关的预测
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        '''
        pred: 模型的预测结果 [batch_size, num_steps, vocab_size] 批次大小、序列长度和词汇表大小
        label: 真实标签
        valid_len: 每个样本的有效长度
        '''
        # 刚开始，所有预测词元的掩码设置为1,形状与标签相同
        weights = torch.ones_like(label)
        # 对weights 将超出valid_len有效长度的部分置为0
        weights = sequence_mask(weights, valid_len)

        # 设置 self.reduction='none' 以获得每个元素的原始损失值
        self.reduction='none'

        # 调用父类方法计算未加权的损失，将预测张量的维度从 (batch_size, num_steps, vocab_size) 调整为 (batch_size, vocab_size, num_steps)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # 将未加权损失与权重相乘得到加权损失，再对时间步维度取平均值
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


loss = MaskedSoftmaxCELoss()
mask_soft_max_loss = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
print('mask_soft_max_loss: ', mask_soft_max_loss)


# 训练
# 特定的序列开始词元（“<bos>”）和 原始的输出序列（不包括序列结束词元“<eos>”） 拼接在一起作为解码器的输入
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        # 使用xavier_uniform_初始化线性层与门控网络
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    # 使用adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    
    # 训练循环
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量

        # 每个批次处理
        for batch in data_iter:
            optimizer.zero_grad()

            # X: 输入序列; X_valid_len: 输入序列的有效长度；Y：目标序列；Y_valid_len: 目标序列的有效长度
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            
            # 为每个样本添加开始标记
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            
            # 将 <bos> 与目标序列的前n-1个元素拼接，形成解码器输入
            dec_input = torch.cat([bos, Y[:, :-1]], 1)

            # 前向传播与计算损失
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1) # 梯度裁剪
            num_tokens = Y_valid_len.sum()
            optimizer.step()

            # 累计损失与处理过的词元数量
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)

        # 每10次迭代输出一次损失与处理速度
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')
    
# 序列到序列的训练过程
# embed_size 嵌入层维度32；num_hiddens：RNN隐藏层单元数32；num_layers：RNN层数2
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
# 每个批次包含样本64，序列最大长度为10个时间步
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

# 加载机器翻译数据集
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

# 编码器 使用源语言词汇表创建
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)

# 解码器 使用目标语言词汇表创建
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)

# 开始训练
net = encoder_decoder_4.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# 预测 通过<bos> <eos>管理每次的预测
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    # 将输入句子转小写并分词，添加序列结束标记
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    # 计算输入序列的有效长度
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # 填充到固定长度
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    
    # 添加批次维度，使形状从 [seq_len] 变为 [1, seq_len]
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    
    # 编码，得到上下文表示
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 初始化解码器状态
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    
    # 初始化解码器输入为开始标记 <bos>
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq, attention_weight_seq = [], []
    # 逐词生成循环
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


# 预测序列的评估函数 当预测序列与标签序列完全相同时，BLEU为1
def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# 利用训练好的循环神经网络“编码器－解码器”模型， 将几个英语句子翻译成法语，并计算BLEU的最终结果
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
