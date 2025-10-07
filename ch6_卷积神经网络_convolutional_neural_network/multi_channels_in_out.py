import torch
from d2l import torch as d2l

# 多输入输出通道

# 多输入单输出通道

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
print(X.shape)

K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(K.shape)

corr2d_multi_in(X, K)


# 多输入多输出通道
# 随着神经网络层数的加深，我们常会增加输出通道的维数，通过减少空间分辨率以获得更大的通道深度
# 直观地说，我们可以将每个通道看作对不同特征的响应
# 因为每个通道不是独立学习的，而是为了共同使用而优化的。因此，多输出通道并不仅是学习多个单通道的检测器


# 计算多个通道的输出的互相关函数
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    # stack 升维拼接
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# 通过将核张量K与K+1（K中每个元素加）和K+2连接起来，构造了一个具有个输出通道的卷积核
K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)

corr2d_multi_in_out(X, K)


# 1*1卷积层 通常用于调整网络层的通道数量
def corr2d_multi_in_out_1x1(X, K):
    # 需要对输入和输出的数据形状进行调整
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
print(X.shape)
K = torch.normal(0, 1, (2, 3, 1, 1))
print(K.shape)

# 当执行1*1卷积运算时，上述函数相当于先前实现的互相关函数 corr2d_multi_in_out
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
