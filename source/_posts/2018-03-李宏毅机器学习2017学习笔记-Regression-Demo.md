---
title: '李宏毅机器学习2017学习笔记:Regression Demo'
categories: Machine Learning
tags:
  - Machine Learning
  - 李宏毅
  - Regression
  - 调参
date: 2018-03-26 19:56:53
updated: 2018-03-26 19:56:53
description: 使用AdaGrad动态调整learning rate.
---


# 视频简介
视频中先固定learning rate，迭代10w次：
1. 首先，设置了一个比较小的learning rate，`lr=1e-6`，没有得到最优解，就停止了；
2. 然后，放大lrarning rate，`lr=1e-5`，结果出现震荡的情况，无法得到最优解。
3. 最后，采用AdaGrad方法调整learning rate，得到了最优解。

# 固定learning rate
```python
lr = 0.000001
for i in range(iteration):
    # calculate gradient
    b_grad = ...
    w_grad = ...

    # update weight
    b = b - lr * b_grad
    w = w - lr * w_ grad
```

# 动态调整learning rate
视频中使用的方法是：AdaGrad

```python
lr = 1
lr_b = 0
lr_w = 0
for i in range(iteration):
    # calculate gradient
    b_grad = ...
    w_grad = ...

    # update lr_b, lr_w with AdaGrad
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2

    # update weight
    b = b - lr/np.sqrt(lr_b) * b_grad
    w = w - lr/np.sqrt(lr_w) * w_ grad
```
