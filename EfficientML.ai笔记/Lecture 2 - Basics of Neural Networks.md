# Lecture 2 - Basics of Neural Networks

## 1.神经网络的基本术语

Batch size：批量大小。是指每次迭代使用的样本数量，影响训练速度，内存消耗等

- neuron：神经元。神经网络的最小计算单位，相当于一个 “加权求和+激活函数”的小模块
- synapses：突触。神经元之间相连的“权重连接”，对应于深度学习中的 `weights` 
- activation：激活。神经元1输出的非线性处理结果，如 `ReLU` , `Sigmoid`

> [!Note]
>
> 意义：引入非线性，让模型能处理复杂关系

- feature：特征。输入或中间层输出的向量，用于表示信息，“神经网络理解世界的方式”

- weight/parameters：权重/参数。模型训练过程中学习和调整的数值，决定模型的行为

## 神经网络的常见构建模块

- Fully-Conneted（全连接层）

​	每个输入和输出都相连，用于结构简单的问题

- Convolution （卷积）

​	用于图像、语言等，局部连接，参数更少

- Grouped Convolution （分组卷积）

​	把通道分成多组，各组单独卷积，减少计算量

- Depthwise Convolution （深度可分离卷积）

​	每个通道分别卷积再混合，非常高效

- Pooling （池化）

​	降低分辨率，减小特征图大小，提高感受野

- Normalization （归一化）

- Transformer （变压器结构）

​	现在最主流的结构，基于attention（注意力机制）

## 神经网络的效率指标

这些指标用于衡量模型的有效性与性能：

- #Parameters（参数量）：模型存储大小。
- Model Size（模型大小）：以 MB、GB 测量。
- Peak #Activations（激活峰值）：训练时内存占用关键指标。
- MAC（乘加运算次数）
- FLOP / FLOPS（浮点运算 / 每秒浮点运算）
- OP / OPS（操作数 / 每秒操作数）
- Latency（延迟）：一次推理需要的时间。
- Throughput（吞吐量）：每秒能处理多少数据（如 1000 推理/秒）。

**这些指标在 AI 算法面向硬件部署特别重要！**