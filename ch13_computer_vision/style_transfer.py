import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 图像风格迁移

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg') # 内容图像
d2l.plt.imshow(content_img);

style_img = d2l.Image.open('../img/autumn-oak.jpg') # 风格图像
d2l.plt.imshow(style_img);

# 预处理与后处理
# 这些数据搬运自VGG模型参数
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    '''
    预处理：对输入图像在RGB三个通道分别做标准化，并将结果变换成卷积神经网络接受的输入格式
    '''
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0) # 加一个维度


def postprocess(img):
    '''
    后处理：将输出图像中的像素值还原回标准化之前的值
    '''
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


# 抽取图像特征
# 使用基于ImageNet数据集预训练的VGG-19模型来抽取图像特征
pretrained_net = torchvision.models.vgg19(pretrained=True)

# 为了抽取图像的内容特征和风格特征，我们可以选择VGG网络中某些层的输出
# 一般来说，越靠近输入层，越容易抽取图像的细节信息 反之，则越容易抽取图像的全局信息
# 1. 为了避免合成图像 过多保留内容图像的细节，我们选择VGG较靠近输出的层，即内容层，来输出图像的内容特征
# 2. 还从VGG中选择不同层的输出来匹配局部和全局的风格，这些图层也称为风格层
# 3. VGG网络使用了5个卷积块。 实验中，我们选择第四卷积块的最后一个卷积层作为内容层，选择每个卷积块的第一个卷积层作为风格层
style_layers, content_layers = [0, 5, 10, 19, 28], [25]


# 使用VGG层抽取特征时，我们只需要用到从输入层到最靠近输出层的内容层或风格层之间的所有层。 下面构建一个新的网络net，它只保留需要用到的VGG的所有层
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])

# 给定输入X，如果我们简单地调用前向传播net(X)，只能获得最后一层的输出。 由于我们还需要中间层的输出
# 因此这里我们逐层计算，并保留内容层和风格层的输出
def extract_features(X, content_layers, style_layers):
    contents = [] # 内容
    styles = [] # 风格
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def get_contents(image_shape, device):
    '''
    对内容图像抽取内容特征
    '''
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    '''
    对风格图像抽取风格特征
    '''
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


# 定义损失函数 它由内容损失、风格损失和全变分损失3部分组成

def content_loss(Y_hat, Y):
    '''
    内容损失 平方误差函数衡量合成图像与内容图像在内容特征上的差异
    Y_hat Y 均为 extract_features 函数计算所得到的内容层的输出
    '''
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()


def gram(X):
    '''
    假设基于风格图像的格拉姆矩阵gram_Y已经预先计算好了
    '''
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    '''
    风格损失的平方误差函数的两个格拉姆矩阵输入分别基于合成图像与风格图像的风格层输出
    '''
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 全变分损失
def tv_loss(Y_hat):
    '''
    全变分损失 像素差的绝对值
    '''
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


# 总的损失函数
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    '''
    风格转移的损失函数是内容损失、风格损失和总变化损失的加权和 
    通过调节这些权重超参数，我们可以权衡合成图像在保留内容、迁移风格以及去噪三方面的相对重要性
    '''
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


# 初始化合成图像
# 在风格迁移中，合成的图像是训练期间唯一需要更新的变量
# 可以定义一个简单的模型SynthesizedImage，并将合成的图像视为模型参数。模型的前向传播只需返回模型参数即可
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight # 这样就可以对图像计算梯度并更新


# 创建了合成图像的模型实例，并将其初始化为图像X
# 风格图像在各个风格层的格拉姆矩阵styles_Y_gram将在训练前预先计算好
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)

    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


# 训练
# 在训练模型进行风格迁移时，我们不断抽取合成图像的内容特征和风格特征，然后计算损失函数
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)

    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))

    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X


device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
