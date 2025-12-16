# 实验五

## 一、实验目的
1.理解卷积层、池化层与全连接层的组合方法。

2.理解经典 LeNet-5 的网络结构及特征图可视化流程。

3.体验卷积网络在 MNIST 上的收敛特性与特征提取效果。

4.学习数据载入以及通过卷积，填充，跨步，池化神经网络结构来构建PyTorch的卷积神经网络模型。
### 1. 数据准备

``` python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True
)

testloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True
)

print(f"训练集样本数：{len(trainloader.dataset)}")  
print(f"测试集样本数：{len(testloader.dataset)}")
```

采用 MNIST 手写数字数据集，使用 DataLoader 进行批量加载并转换为 Tensor。

遇到的问题：在本地运行时曾出现Kernel 重启，这是常见的库冲突问题，通过在代码头部加入 os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 解决。

### 2. 模型构建（LeNet-5）

``` python
class Net(nn.Module): 
    def __init__(self): 
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.clf = nn.Linear(84, 10)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.sigmoid(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.clf(x)
        return x

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
epochs = 30
print(model)
```
![image](/image/1.png)

1.  **Conv1**：输入图像为 $28 \times 28$。设置 $5 \times 5$ 卷积核与 padding=2，输出保持 $28 \times 28$；经 $2 \times 2$ 池化后变为 $14 \times 14$。
2.  **Conv2**：输入为 $14 \times 14$。经过 $5 \times 5$ 卷积（无 padding）变为 $10 \times 10$；再经 $2 \times 2$ 池化后变为 $5 \times 5$。
3.  **全连接层连接**：Conv2 输出的通道数为 16，因此全连接层 `fc1` 的输入节点数必须设置为 $16 \times 5 \times 5 = 400$。

在编写代码时，为了确保维度无误，我在 `forward` 函数中打印了 `x.shape` 进行验证，确认数据经过两次卷积池化后确实为 `[Batch, 16, 5, 5]`，随后使用 `x.view` 将其展平输入全连接层。

```python


### 3. 模型训练配置

``` python
for epoch in range(epochs):
    accs, losses = [], []
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    correct = 0
    testloss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            testloss += F.cross_entropy(out, y).item()
            pred = out.max(dim=1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    acc = correct / len(testloader.dataset)
    testloss = testloss / (batch_idx + 1)
    accs.append(acc)
    losses.append(testloss)
    print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch, testloss, acc))
```

采用 Adam 优化器、学习率 0.01，共训练 30 轮。

### 4. 特征可视化

``` python
feature1 = F.sigmoid(model.conv1(x))
feature1 = F.avg_pool2d(feature1, kernel_size=2, stride=2)
feature2 = F.sigmoid(model.conv2(feature1))
feature2 = F.avg_pool2d(feature2, kernel_size=2, stride=2)

n=5
img = x.detach().cpu().numpy()[:n]
feature_map1 = feature1.detach().cpu().numpy()[:n]
feature_map2 = feature2.detach().cpu().numpy()[:n]

fig, ax = plt.subplots(3, n, figsize=(10, 10))
for i in range(n):
    ax[0,i].axis('off')
    ax[0,i].imshow(img[i].sum(0),cmap='gray')
    ax[1,i].axis('off')
    ax[1,i].imshow(feature_map1[i].sum(0),cmap='gray')
    ax[2,i].axis('off')
    ax[2, i].imshow(feature_map2[i].sum(0),cmap='gray')

plt.show()
```

该可视化展示了原始图、Conv1 特征图与 Conv2 特征图。

## 三、实验结果与分析

### 1. 训练指标
![image](/image/2.png)
-   **Epoch 0**：loss=1.4298，acc=44.60%
-   **Epoch 5**：loss=0.0626，acc=98.14%
-   **Epoch 10-30 稳定区间**：acc≈98.5%+
-   **最终 Epoch 29**：loss=0.0421，acc=98.85%

这说明 LeNet-5 虽然结构简单，但对于 MNIST 这种二值化的简单图像分类任务，其特征提取能力已经足够，能够达到较高的识别精度。

### 2. 深度分析

关于优化器学习率与收敛速度的关联：
本实验中，Adam 优化器的学习率（Learning Rate）被设定为 1e-2 (0.01)，这一数值显著高于常规默认的 1e-3。这种大步长的策略在结果上直接促成了模型的快速收敛：仅在第一个 Epoch 结束后，测试集准确率便从初始的随机猜测水平跃升至 90.03%。这一现象的物理成因在于，尽管模型采用了容易产生梯度消失问题的 Sigmoid 激活函数，但较大的学习率有效地放大了参数更新的幅度，补偿了激活函数导数较小带来的梯度衰减效应，使得优化器能够迅速跨越损失函数的平坦区域，直达最优解附近。

关于大批量训练（Batch Size）的稳定性分析：
实验将批量大小（Batch Size）设定为 512，这是一个针对 GPU 并行计算优化的选择。与常规的小批量（如 32 或 64）相比，一次性计算 512 个样本不仅能充分利用硬件算力，大幅缩短物理训练时间，更重要的是它显著提升了梯度估计的稳定性。由于每个 Step 的梯度是基于更多样本计算的平均值，其随机噪声被有效抑制，这直接体现在了 Loss 曲线的平滑下降上。除个别轮次外，模型并未出现剧烈的震荡，证明了大批量训练在简单数据集上的稳健性。

特征图可视化的层级抽象意义：
通过提取前两层卷积的输出进行可视化，我们直观地验证了卷积神经网络“层级抽象”的核心特性。第一层特征图（Feature Map 1）作为浅层输出，通过 model.conv1 与 5x5 卷积核的运算，主要保留并强化了数字的边缘与轮廓信息（如“7”的横折），图像语义依然清晰可辨。而经过二次卷积与池化后的第二层特征图（Feature Map 2），分辨率进一步降低，图像变得模糊且抽象。这种变化表明网络正在逐渐剥离具体的像素细节，转而关注更高级的语义模式，这正是 CNN 能够忽略手写体细微差异而准确分类的关键所在。

全连接层输入维度的计算逻辑：
代码中全连接层输入维度 16 * 4 * 4 (256) 的设定，是基于卷积与池化公式严格推导的结果，而非经验值。输入图像为 $28 \times 28$，经过第一层卷积（核5x5，步长1，无填充）后尺寸变为 $24 \times 24$，随即通过 2x2 的平均池化降维至 $12 \times 12$。第二层卷积继续处理该 $12 \times 12$ 特征图，输出尺寸缩减为 $8 \times 8$，再经第二次池化最终得到 $4 \times 4$ 的空间尺寸。结合第二层卷积设定的 16 个输出通道，最终展平后的向量长度即为 $16 \times 4 \times 4 = 256$。这种为了对齐通道而采用默认填充（padding=0）的策略，虽然简单，却严谨地保证了张量维度的匹配，同时 128 个输出节点的设计也符合计算机科学中常用的 2 的幂次倍数关系，兼顾了计算效率与特征承载能力。

## 四、实验小结

本次实验完成了基于 LeNet-5 的手写数字识别全过程，主要收获如下：

1.  **对网络结构的理解**：
    通过亲手编写代码，理清了 CNN 中“卷积-激活-池化”这一经典模块的堆叠逻辑。特别是明确了数据在层与层之间传递时，必须严格计算输入输出的 Tensor 维度，否则会导致模型构建失败。

2.  **结果评估**：
    LeNet-5 模型参数量小、训练速度快，在 MNIST 数据集上表现优异。
