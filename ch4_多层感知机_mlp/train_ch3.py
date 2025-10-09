import torch.nn
from d2l import torch as d2l
from IPython import display
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """初始化动画"""
        if legend is None:
            legend = []

        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel,
                                                xlim, ylim, xscale, yscale,
                                                legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y): #添加数据点
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        import matplotlib.pyplot as plt
        plt.savefig(f'./epoch_{len(self.X[0])}.png')  # 每个epoch保存一张图片

        display.display(self.fig)
        display.clear_output(wait=True)  #清除输出，等待下一个输出
        
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) #axis=1表示按行取最大值
    cmp = y_hat.type(y.dtype) == y #将y_hat和y转换为相同的数据类型，然后比较是否相等
    return float(cmp.type(y.dtype).sum()) #将布尔值转换为浮点数，然后求和
#type(y.dtype)表示将y转换为相同的数据类型
        
class Accumulator:
    """在多个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n  #初始化一个长度为n的列表，元素为0.0

    def add(self, *args):
        """添加一个或多个值"""
        self.data = [a + float(b) for a, b in zip(self.data, args)]  #将每个元素与对应的参数相加
        
    def reset(self):
        """将所有元素重置为0"""
        self.data = [0.0] * len(self.data)  #将列表中的所有元素重置为0

    def __getitem__(self, idx):
        """获取指定索引的值"""
        return self.data[idx]  #返回指定索引的值

def evaluate_accuracy(net, data_iter): #net是模型，data_iter是数据迭代器
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  #将模型设置为评估模式
    metric = Accumulator(2)  #用于计算两个值的累加器
    with torch.no_grad(): #关闭梯度计算
        for X, y in data_iter: #遍历数据迭代器
            metric.add(accuracy(net(X), y), y.numel()) #计算准确率和样本数量,numel()返回张量的元素数量
    return metric[0] / metric[1] #返回准确率


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module): #判断net是否为torch.nn.Module的实例
        net.train()  #将模型设置为训练模式
    metric = Accumulator(3)  #用于计算三个值的累加器
    for X, y in train_iter:
        y_hat = net(X)  #前向传播
        l = loss(y_hat, y)  #计算损失
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()  #清除梯度
            l.mean().backward()  #反向传播 当使用pytorch优化器时
            updater.step()  #更新参数
        else:
            l.sum().backward()  #反向传播 使用自定义损失函数
            updater(X.shape[0])  #更新参数
        metric.add(l.sum(), accuracy(y_hat, y), y.numel())  #累加损失、准确率和样本数量
    return metric[0] / metric[2], metric[1] / metric[2]  #返回平均损失和平均准确率


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', ylabel='loss',
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  #训练一个迭代周期
        test_acc = evaluate_accuracy(net, test_iter)  #计算测试集上的精度
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc #assert语句用于测试条件是否为真，如果不为真则抛出异常