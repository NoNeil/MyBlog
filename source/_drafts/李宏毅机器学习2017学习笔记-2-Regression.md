---
title: '李宏毅机器学习2017学习笔记:2.Regression'
date: 2018-03-24 17:52:57
updated: 2018-03-24 17:52:57
categories: Machine Learning
description: 摘要
tags: 
- 李宏毅
- Machine Learning
- Regression
- LR
---


<!-- more --> 

问题描述：
如何根据宝可梦的CP值预测进化后的CP值？

建模：
$$y = b + w * X_{cp}$$

损失函数：
$$L(f) = L(w, b) = \frac12 \sum_{i=1}^{10} \left(\hat {y}^i - (b +  w · x_{cp}^i)\right) ^2$$

优化方法（梯度下降法）：

$$ f^\ast = arg \min_{f}^{} L(f) $$
$$ w^\ast, b^\ast = arg \min_{w, b} \frac12 \sum_{i=1}^{10}\left(\hat {y}^i - (b +  w · x_{cp}^i)\right)^2$$

梯度：
$$ \frac {\partial L} {\partial w} = \sum_{i=1}^{10} \left( \hat {y}^i - (b +  w · x_{cp}^i)\right) (-x_{cp}^i)$$

$$ \frac {\partial L} {\partial b} = \sum_{i=1}^{10}\left(\hat {y}^i - (b +  w · x_{cp}^i)\right) (-1)$$

正则化（Regularization）：


