# 基于 LeNet-5 的手写数字识别实验报告

## 一、实验目的

掌握卷积神经网络（CNN）基本原理，明确卷积层、池化层（下采样）和全连接层的作用与工作机制。熟悉
PyTorch 深度学习框架，学会运用 `torch.nn`
构建网络模型、`torch.utils.data` 加载数据集及 `optimizer`
进行模型优化。在 MNIST 数据集上复现并训练 LeNet-5
网络完成手写数字识别任务。通过可视化卷积层中间特征图（Feature
Maps），直观理解神经网络提取特征的过程。

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

模型包含 2 个卷积层与 3 个全连接层，结构完整复现 LeNet-5。

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

模型收敛迅速，精度接近 99%，符合 LeNet-5 在 MNIST 数据集上的典型表现。

### 2. 特征图分析
![image](/image/3.png)
-   **第一层卷积**：保留边缘信息，图像略模糊但轮廓明显。
-   **第二层卷积**：分辨率降至 5×5，图像抽象化，体现高级特征。

## 四、实验小结

本实验基于 PyTorch 实现并训练了 LeNet-5 模型，测试集准确率达到
**98.85%**。从可视化结果中可清晰观察卷积层如何逐层提取特征，加深了对 CNN
结构的理解。虽然使用 Sigmoid 和 AvgPool（较经典的配置），但该结构对
MNIST 依然足够有效。未来可尝试 ReLU、MaxPool、学习率衰减或 Dropout
来进一步提升模型性能。
