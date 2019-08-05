# DANet-keras
keras-[Dual Attention Network for Scene Segmentation](<https://arxiv.org/abs/1809.02983>)

## Abstract

提出了双重注意网络（DANet）来自适应地集成局部特征和全局依赖。在传统的扩张FCN之上附加两种类型的注意力模块，分别模拟空间和通道维度中的语义相互依赖性。

- 位置注意力模块通过所有位置处的特征的加权和来选择性地聚合每个位置的特征。无论距离如何，类似的特征都将彼此相关。
- 同通道注意力模块通过整合所有通道映射之间的相关特征来选择性地强调存在相互依赖的通道映射。
- 将两个注意模块的输出相加以进一步改进特征表示，这有助于更精确的分割结果

## 1. Introduction

- 提出了一个双重注意网络（DANet）来捕捉空间和通道维度中的全局特征依赖关系
- 建议使用位置注意力模块来学习特征的空间相互依赖性，并设计通道注意力模块来模拟通道相互依赖性。通过在局部特征上建模丰富的上下文依赖关系，显著改善了分割结果
- 在三个数据集Cityscapes， PASCAL Context和COCO Stuff上实现了state-of-the-art的结果

## 2. Dual Attention Network

### 2.1 Overview

![img](C:\Users\CV\Documents\GitHub\DANet-keras\readme_img\network——overciew.png)

- 采用 Dilated ResNet (DeepLab V2的主干网络)作为主干，删除了下采样操作并在最后两个ResNet块中使用了空洞卷积，最终特征映射的大小是输入图像的1/8
- Dilated ResNet 产生的特征图再送入两个平行的注意力模块中
- 最后汇总两个注意力模块的输出特征，以获得更好的像素级预测特征表示

### 2.2 Position Attention Module

![img](C:\Users\CV\Documents\GitHub\DANet-keras\readme_img\pam.png)

- 特征图**A**(C×H×W)首先分别通过3个卷积层得到3个特征图**B,C,D,**然后将**B,C,D** reshape为C×N，其中N=H×W
- 之后将reshape后的**B**的转置(NxC)与reshape后的C(CxN)相乘，再通过softmax得到spatial attention map **S**(N×N)
- 接着在reshape后的**D**(CxN)和**S**的转置(NxN)之间执行矩阵乘法，再乘以尺度系数α，再reshape为原来形状，最后与**A**相加得到最后的输出**E**
- 其中α初始化为0，并逐渐的学习得到更大的权重

### 2.3 Channel Attention Module

![img](C:\Users\CV\Documents\GitHub\DANet-keras\readme_img\cam.png)

- 分别对**A**做reshape(CxN)和reshape与transpose(NxC)
- 将得到的两个特征图相乘，再通过softmax得到channel attention map **X**(C×C)
- 接着把**X**的转置(CxC)与reshape的**A**(CxN)做矩阵乘法，再乘以尺度系数β，再reshape为原来形状，最后与**A**相加得到最后的输出**E**
- 其中β初始化为0，并逐渐的学习得到更大的权重

### 2.4 Attention Module Embedding with Networks

- 两个注意力模块的输出先进行元素求和以完成特征融合
- 再进行一次卷积生成最终预测图

## 3. 实验设置

我们的实现基于Pytorch。采用poly 学习率策略，其中初始学习率在每次迭代之后乘以$(1-iter/total_iter)^{0.9}$。Cityscapes数据集的基本学习率设置为0.01。动量衰减系数为0.9，重量衰减系数为0.0001。批处理大小对Cityscapes设置为8，对其他数据集设置为16。在采用多尺度增强时，将COCO的训练时间设置为180个epoch，其他数据集的训练时间设置为240个epoch。当使用两个注意模块时，我们在网络的顶端采用多个损失函数。为了增加数据，我们在Cityscapes数据集的消融研究中采用随机裁剪(裁剪大小768)和随机左右翻转。