import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 全卷积网络 fully convolutional network，FCN
# 全卷积网络先使用卷积神经网络抽取图像特征，然后通过1*1卷积层将通道数变换为类别个数
# 最后在通过转置卷积将特征图的高和宽变换为输入图像的尺寸。 因此，模型输出与输入图像的高和宽相同，且最终输出通道包含了该空间位置像素的类别预测

pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:] # 查看模型最后3层
#  AdaptiveAvgPool2d(output_size=(1, 1)), # 倒数第二层全局平均汇聚层
#  Linear(in_features=512, out_features=1000, bias=True)] # 最后一层 MLP


# 创建一个全卷积网络net]。 它复制了ResNet-18中大部分的预训练层，除了最后的全局平均汇聚层和最接近输出的全连接层
net = nn.Sequential(*list(pretrained_net.children())[:-2])

# 给定高度为320和宽度为480的输入，net的前向传播将输入的高和宽减小至原来的，即10和15
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape # 输出为[1, 512, 10, 15]

# 接下来[使用1*1卷积层将输出通道数转换为Pascal VOC2012数据集的类数（21类）。] 
# 最后需要(将特征图的高度和宽度增加32倍)，从而将其变回输入图像的高和宽
# 由于(320-64+16*2+32)/32=10 和(480-64+16*2+32)/32=15 我们构造一个步幅为32的转置卷积层,并将卷积核的高和宽设为64，填充为16

# 我们可以看到如果步幅为s, 填充为s/2(s/2假设是整数),且卷积核的高和宽为2s,转置卷积核会将输入的高和宽分别放大s倍
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1)) # 1*1卷积层，更新输出通道数
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32)) # 转置卷积层，将特征图放大32倍

# 初始化转置卷积层
def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    用双线性插值初始化转置卷积层，加快收敛速度(也可以随机初始化)
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

# 构造一个将输入的高和宽放大2倍的转置卷积层，并将其卷积核用bilinear_kernel函数初始化
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));

# 验证是都变成2倍
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()

d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);

# 全卷积网络[用双线性插值的上采样初始化转置卷积层，对于1*1卷积层，使用Xavier初始化参数]
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);


# =========================
# 读取数据集 指定随机裁剪的输出图像的形状为320*480 能被32整除
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

# 训练
def loss(inputs, targets):
    # 基于每个像素求损失
    # 原数据是2维，做一次平均变成1维度，在作一次平均变成标量
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# 预测 在预测时，我们需要将输入图像在各个通道做标准化，并转成卷积神经网络所需要的四维输入格式
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0) # 增加一个维度 batch_size=1
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


# 可视化预测的类别 给每个像素，我们将预测类别映射回它们在数据集中的标注颜色
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]


# 为简单起见，我们只读取几张较大的测试图像，并从图像的左上角开始截取形状为320*480的区域用于预测
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
