# Lab 0 - PyTorch Tutorial

[lab链接]([Lab0.ipynb - Colab](https://colab.research.google.com/drive/1gvxq7mIAeIBAtmLKH1Q1GknA-GRsK7Q6#scrollTo=inwZEfX3Mo6A))

In this tutorial, we will explore how to train a neural network with PyTorch.

>把colab上的文档整理成中文便于查阅，添加了必要的代码注释。

[PyTorch官方中文文档](https://docs.pytorch.ac.cn/docs/stable/index.html)

## 环境配置

### 安装必要的packages

~~~~bash
!pip install torchprofile 1>/dev/null
~~~~

- 环境中安装 pytorch
-  `1>/dev/null` 表示将输出信息重定向到空设备中，即隐藏输出信息，但如果有错误仍会显示

>colab中，`!...` 表示该语句是在终端执行的语句，而没有感叹号的时候，表示是执行 `python` 代码语句 

### 导入库

~~~~python
import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
~~~~

### 随机数种子

为确保可重复性，我们将控制随机生成器的种子

~~~~python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
~~~~

>可重复性体现在：这里的随机数是伪随机，每次随机产生的结果是一致的，那么

## 数据预处理

在本教程中，我们将使用CIFAR-10作为目标数据集。该数据集包含来自10个类别的图像，每张图像的大小为3x32x32，即3通道彩色图像，尺寸为32x32像素。

~~~python
"""数据变换定义：train：训练集；test：测试集"""
transforms = {
  "train": Compose([
    RandomCrop(32, padding=4),#随机裁剪+填充
    RandomHorizontalFlip(),#随机水平翻转
    ToTensor(),#转换为Tensor
  ]),
  "test": ToTensor(),#测试集只做Tensor转换
}
"""数据集加载"""
dataset = {}
for split in ["train", "test"]:
  dataset[split] = CIFAR10(
    root="data/cifar10",#数据存储路径
    train=(split == "train"),#True=训练集，False=测试集
    download=True,#如果不存在则下载
    transform=transforms[split],#应用对应的变换
  )
~~~

### 训练集变换的作用

- `RandomCrop(32, padding=4)` - 数据增强，提高模型泛化能力
    - 先填充4像素，再随机裁剪回32x32
    - 让模型学习不同位置的物体
- `RandomHorizontalFlip()` - 随机水平翻转，模拟镜像图像
- `ToTensor()` - 将PIL图像转为PyTorch Tensor，并归一化到[0,1]

测试集不需要数据增强0

### 数据集结构

- **训练集**: 50,000张32x32彩色图像
- **测试集**: 10,000张32x32彩色图像
- **类别**: 10个类别（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）

最终得到的 dataset 字典：

~~~python
dataset["train"] #训练数据集（带数据增强）
dataset["test"] #测试数据集（无数据增强）
~~~

### 可视化数据集

我们可以可视化数据集中的一些图像以及其对应的类别标签

~~~python
"""样本收集"""
samples = [[] for _ in range(10)] #创建10个空列表，对应10个类别
for image, label in dataset["test"]:
  if len(samples[label]) < 4:	#每个类别最多收集4张
    samples[label].append(image)	#存储图像张量

plt.figure(figsize=(20, 9))# 创建20英寸宽、9英寸高的画布
for index in range(40):	#总共40张图片（10类别*4张）
  label = index % 10	#计算类别索引（0-9）
  image = samples[label][index // 10]	#获取对应图像

  # Convert from CHW to HWC for visualization（从CHW转为HWC格式）
  image = image.permute(1, 2, 0)

  # Convert from class index to class name
  label = dataset["test"].classes[label]	#数字转文字标签

  # Visualize the image
  plt.subplot(4, 10, index + 1)
  plt.imshow(image)
  plt.title(label)
  plt.axis("off")
plt.show()
~~~

### 分批处理

为了训练一个神经网络，我们需要分批输入数据。我们创建批次大下为512的数据加载器

~~~python
dataflow = {}
for split in ['train', 'test']:
  dataflow[split] = DataLoader(
    dataset[split],
    batch_size=512,
    shuffle=(split == 'train'),
    num_workers=0,
    pin_memory=True,
  )
~~~

我们可以输出数据加载器中的数据形式和形状

~~~python
for inputs, targets in dataflow["train"]:
  print("[inputs] dtype: {}, shape: {}".format(inputs.dtype, inputs.shape))
  print("[targets] dtype: {}, shape: {}".format(targets.dtype, targets.shape))
  break
~~~

## 模型

在本教程中，我们将使用 VGG-11 的一个变体（下采样次数“downsamples”更少且分类器更小）作为我们的模型。

>VGG是经典的CNN架构，以其简单的重复结构著称

### 模型构建

#### 1.架构定义

~~~python
ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
~~~

这定义了VGG的层结构：

- **数字**：卷积层的输出通道数
- **'M'**：最大池化层(MaxPool)
- 对应VGG11的变体

（这里还是不太清楚，ai给出的，后续得查阅一下）

#### 2.层构建逻辑

~~~python
def add(name: str, layer: nn.Module) -> None:
    layers.append((f"{name}{counts[name]}", layer))
    counts[name] += 1
~~~

这个辅助函数用于：

- 自动给层命名（conv0, conv1, bn0, bn1...）
- 维护每种层的计数

#### 3.主干网络构建

~~~python
in_channels = 3  # 输入通道数(RGB)
for x in self.ARCH:
    if x != 'M':
        # 卷积块: Conv2d -> BatchNorm -> ReLU
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))	#这里用到BatchNorm
        add("relu", nn.ReLU(True))
        in_channels = x  # 更新输入通道数
    else:
        # 池化层: 2x2 MaxPool
        add("pool", nn.MaxPool2d(2))
~~~

#### 4.网络组件

~~~python
self.backbone = nn.Sequential(OrderedDict(layers))  # 特征提取主干
self.classifier = nn.Linear(512, 10)  # 分类器(10个类别)
~~~

#### 5.卷积层配置

~~~python
nn.Conv2d(in_channels, x, 3, padding=1, bias=False)
x = x.mean([2, 3])  # 对高度和宽度维度求平均
~~~

- **3x3卷积核**：VGG的标准选择
- **padding=1**：保持特征图尺寸不变
- **bias=False**：因为后面接BatchNorm，可以省略偏置

#### 6.模型实例化

~~~python
model = VGG().cuda()  # 创建模型并移动到GPU
~~~

### 模型检查

#### 1.输出主干结构

其骨干部分由八个“卷积 - 归一化 - 激活”模块组成，其间穿插着四个最大池化层，以将特征图的尺寸缩小 2 的 4 次方即 16 倍：

~~~python
print(model.backbone)
~~~

#### 2.输出分类器部分

在特征图经过池化操作之后，其分类器通过一个线性层来预测最终的输出

~~~python
print(model.classifier)
~~~

#### 3.效率检验

由于本课程注重效率，接下来我们将检查其模型大小以及（理论上的）计算成本。

- 模型大小可以通过可训练参数的数量来估计：

~~~python
num_params = 0
for param in model.parameters():
  if param.requires_grad:
    num_params += param.numel()
print("#Params:", num_params)
~~~

- 计算成本可通过 TorchProfile 统计的乘累加操作数（MACs）进行估算：

~~~python
num_macs = profile_macs(model, torch.zeros(1, 3, 32, 32).cuda())
print("#MACs:", num_macs)
~~~

该模型包含 **9.2M 参数**，推理时需执行 **606M 次乘累加操作（MACs）**。我们将在后续几个实验环节中共同优化其效率。

## 优化

### 损失函数

由于我们当前处理的是分类问题，将采用**交叉熵（Cross Entropy）**作为损失函数来优化模型：

~~~python
criterion = nn.CrossEntropyLoss()
~~~

### 优化过程

优化过程将采用**带动量的随机梯度下降（Stochastic Gradient Descent, SGD）**进行：

（还不知道是啥）

~~~~python
optimizer = SGD(
  model.parameters(),
  lr=0.4,
  momentum=0.9,
  weight_decay=5e-4,
)
~~~~

学习率将采用以下调度器（改编自[本系列博客](https://www.google.com/url?q=https%3A%2F%2Fmyrtle.ai%2Flearn%2Fhow-to-train-your-resnet%2F)）进行动态调节：

（似乎找不到这个网页）

~~~python
num_epochs = 20
steps_per_epoch = len(dataflow["train"])

# Define the piecewise linear scheduler
lr_lambda = lambda step: np.interp(
  [step / steps_per_epoch],
  [0, num_epochs * 0.3, num_epochs],
  [0, 1, 0]
)[0]

# Visualize the learning rate schedule
steps = np.arange(steps_per_epoch * num_epochs)
plt.plot(steps, [lr_lambda(step) * 0.4 for step in steps])
plt.xlabel("Number of Steps")
plt.ylabel("Learning Rate")
plt.grid("on")
plt.show()

scheduler = LambdaLR(optimizer, lr_lambda)
~~~

## 训练

### 定义单循环函数

我们首先定义**单轮训练函数**（*即完整遍历一次训练集*）用于模型优化：

~~~python
def train(
  model: nn.Module,
  dataflow: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  scheduler: LambdaLR,
) -> None:
  model.train()

  for inputs, targets in tqdm(dataflow, desc='train', leave=False):
    # Move the data from CPU to GPU
    inputs = inputs.cuda()
    targets = targets.cuda()

    # Reset the gradients (from the last iteration)
    optimizer.zero_grad()

    # Forward inference
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward propagation
    loss.backward()

    # Update optimizer and LR scheduler
    optimizer.step()
    scheduler.step()
~~~

### 定义评估函数

随后定义**评估函数**，用于在测试集上计算评估指标（*本任务中特指准确率*）：

~~~python
@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataflow: DataLoader
) -> float:
  model.eval()

  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataflow, desc="eval", leave=False):
    # Move the data from CPU to GPU
    inputs = inputs.cuda()
    targets = targets.cuda()

    # Inference
    outputs = model(inputs)

    # Convert logits to class indices
    outputs = outputs.argmax(dim=1)

    # Update metrics
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  return (num_correct / num_samples * 100).item()
~~~

### 启动训练

在定义完**训练函数**和**评估函数**后，即可启动模型训练！预计耗时约 10 分钟。

~~~python
for epoch_num in tqdm(range(1, num_epochs + 1)):
  train(model, dataflow["train"], criterion, optimizer, scheduler)
  metric = evaluate(model, dataflow["test"])
  print(f"epoch {epoch_num}:", metric)
~~~

如果一切顺利，训练的模型应该有大于92.5%的准确性

## 可视化

我们可以通过**可视化模型预测结果**来直观评估其真实性能：

~~~python
plt.figure(figsize=(20, 10))
for index in range(40):
  image, label = dataset["test"][index]

  # Model inference
  model.eval()
  with torch.inference_mode():
    pred = model(image.unsqueeze(dim=0).cuda())
    pred = pred.argmax(dim=1)

  # Convert from CHW to HWC for visualization
  image = image.permute(1, 2, 0)

  # Convert from class indices to class names
  pred = dataset["test"].classes[pred]
  label = dataset["test"].classes[label]

  # Visualize the image
  plt.subplot(4, 10, index + 1)
  plt.imshow(image)
  plt.title(f"pred: {pred}" + "\n" + f"label: {label}")
  plt.axis("off")
plt.show()
~~~



