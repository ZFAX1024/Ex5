# 实验五

## 一、实验目的

本次实验旨在通过手写 LeNet-5 网络，在代码层面理清 CNN 的数据流向，特别是卷积层到全连接层时的维度变化。同时，利用 PyTorch 的 Hook 或中间层提取机制，可视化卷积核提取到的 Feature Maps，从直观上验证卷积操作对边缘和纹理的提取能力。

## 二、实验内容

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

## 四、实验小结

本次实验完成了基于 LeNet-5 的手写数字识别全过程，主要收获如下：

1.  **对网络结构的理解**：
    通过亲手编写代码，理清了 CNN 中“卷积-激活-池化”这一经典模块的堆叠逻辑。特别是明确了数据在层与层之间传递时，必须严格计算输入输出的 Tensor 维度，否则会导致模型构建失败。

2.  **结果评估**：
    LeNet-5 模型参数量小、训练速度快，在 MNIST 数据集上表现优异。
