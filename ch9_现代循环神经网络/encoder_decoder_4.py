from torch import nn

# 编码器-解码器接口
# encoder 编码器:接受一个长度可变的序列作为输入， 并将其转换为具有固定形状的编码状态
# decoder 解码器:将固定形状的编码状态映射到长度可变的序列

# 编码器和解码器结构只是一种神经网络的抽象结构适用于很多网络，他两不一定需要同一类型的神经网络

#@save
class Encoder(nn.Module):
    """基本编码器接口"""
    def __init__(self, **kwargs): # **kwargs 接收任意关键字参数
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        # 接受输入张量X和可选的额外参数(*args)
        # 定义为抽象方法，必须在具体的子类种重写实现
        raise NotImplementedError


#@save
class Decoder(nn.Module):
    """基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    # 将编码器的输出（enc_outputs）转换为编码后的状态
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


# 合并编码器与解码器
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
