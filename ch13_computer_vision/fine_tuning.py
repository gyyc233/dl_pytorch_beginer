import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 模型微调 fine tuning

# 下载热狗数据集

#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')

# 分别读取训练和测试数据集
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 显示了前8个正类样本图片和最后8张负类样本图片
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);

# 在训练期间，我们首先从图像中裁切随机大小和随机长宽比的区域，然后将该区域缩放为(224*224)输入图像
# 在测试过程中，我们将图像的高度和宽度都缩放到256像素然后裁剪中央224*224作为输入

# 此外，对于RGB（红、绿和蓝）颜色通道，我们分别标准化每个通道。 具体而言，该通道的每个值减去该通道的平均值，然后将结果除以该通道的标准差

# 使用ImageNet数据集的统计值来标准化RGB图像 [0.485, 0.456, 0.406] 是RGB三通道的均值, [0.229, 0.224, 0.225] 是RGB三通道的标准差
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224), # 随机裁剪并调整图像大小至224×224像素
    torchvision.transforms.RandomHorizontalFlip(), # 随机水平翻转
    torchvision.transforms.ToTensor(), # 将PIL图像转换为PyTorch张量，并将像素值缩放到[0,1]范围
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

# 使用预训练模型 ResNet-18
pretrained_net = torchvision.models.resnet18(pretrained=True)

# 预训练的源模型实例包含许多特征层和一个输出层fc pretrained_net.fc
# Linear(in_features=512, out_features=1000, bias=True)

# 此划分的主要目的是促进对除输出层以外所有层的模型参数进行微调
# 在ResNet的全局平均汇聚层后，全连接层转换为ImageNet数据集的1000个类输出，我们定义一个系的输出层，它的定义方式与预训练源模型的定义方式相同，只是最终层中的输出数量被设置为目标数据集中的类数

finetune_net = torchvision.models.resnet18(pretrained=True)
# 定义新的输出层并使用xavier初始化
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2) 
nn.init.xavier_uniform_(finetune_net.fc.weight)

# 定义一个训练函数进行微调
def train_fine_tuning(net,learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = torch.utils.data.DataLoader(
        train_imgs, transform=train_augs, # transform=train_augs: 指定了数据预处理和增强操作
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        test_imgs, transform=test_augs, batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')

    # fine tuning
    if param_group: # 参数分组,是否对模型的不同层使用不同的学习率
        # 1. 从模型中返回所有参数的名称与参数本身 for name,param in net.named_parameters() 如 ("conv1.weight", tensor), ("bn1.weight", tensor), ("fc.weight", tensor)
        # 2. 如果名字不是"fc.weight","fc.bias"则提取到params中
        # 3. params_lx包含了模型中除最后分类层外的所有参数
        params_lx=[param for name,param in net.named_parameters() if name not in ["fc.weight","fc.bias"]]

        # 对模型的不同参数组使用不同学习率
        # 第一组参数 {'params': params_lx} 使用默认学习率与L2正则化系数
        # 第二组参数 {'params': net.fc.parameters(), 'lr': learning_rate * 10} 使用较高的学习率，使用L2正则化系数
        trainer=torch.optim.SGD([{'params': params_lx},{'params':net.fc.parameters(),'lr':learning_rate*10}],lr=learning_rate,weight_decay=0.001)
    else:
        trainer=torch.optim.SGD(net.parameters(),lr=learning_rate,weight_decay=0.001)

    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


# 使用较小的学习率，通过微调预训练获得更新后的模型参数
train_fine_tuning(finetune_net, 5e-5)

# 进行比较 定义了一个相同的模型，但是将其所有模型参数初始化为随机值（不适用预训练），由于整个模型需要从头开始训练，因此我们需要使用更大的学习率
scratch_net = torchvision.models.resnet18(pretrained=False)
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
