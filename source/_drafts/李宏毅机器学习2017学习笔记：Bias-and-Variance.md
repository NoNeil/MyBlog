---
title: 李宏毅机器学习2017学习笔记：Bias and Variance
date: 2018-03-26 20:29:58
updated: 2018-03-26 20:29:58
description: 如何分析error来自哪里？进而如何调整模型？
categories: Machine Learning
tags:
- Machine Learning
- 李宏毅
- Bias
- Variance
---

# Error来自哪里？
* 来自于bias
* 来自于variance


# Estimator(估计量)
在估计宝可梦的CP值的例子中，正确的函数为 $\hat f$ ，这个我们无法知道。

我们只能从训练数据中学到一个最好的函数 $f^\ast$ .所以，$f^\ast$ 是 $\hat f$的一个estimator。

## 『估计量』 的 Bia 和 Variance
假设有一个变量$x$，满足：$x$的均值为$\mu$，均方差为$\sigma^2$。

### 如何估计均值$\mu$呢？
取`N`个点：${x^1, x^2,...,x^N}$
`N`个点取均值，结果不会等于$\mu$，当`N`无限大时，均值会无限接近$\mu$：
$$ m = \frac 1 N \sum_n x^n \neq \mu $$
虽然每个$m$与$\mu$都不相等，但是$m$的期望等于均值$\mu$。
所以用$m$来估计$\mu$，是**无偏**的。
$$ E[m] = E \left[\frac 1 N \sum_n x^n \right] = \frac 1 N \sum_n E[x^n] = \mu $$

就像打靶的时候，瞄的点是$\mu$，但是由于风、肌肉抖动等的影响，实际打中的地方会散布在瞄的$\mu$的周围。

散布得多散，取决于$m$的方差：
$$ Var(m) = \frac {\sigma^2} N $$

$m$的方差取决于样本的数量：
* 当`N`比较小时，散布比较开；
* 当`N`比较大时，散布比较紧。
<img src="./bias_of_estimator.jpg" width="500px">

### 如何估计均方差$\sigma^2$呢？
$s^2$表示：
$$ s^2 = \frac 1 N \sum_n (x^n - m)^2 $$
用$s^2$来估计均方差，是**有偏**的：
$$ E[s^2] = \frac {N-1} N \sigma^2 \neq \sigma^2 $$
当`N`很大时，$s^2$的期望会很接近$\sigma^2$.
<img src="./variance_of_estimator.jpg" width="500px">

### 小结
我们的目标是估测靶的中心$\hat f$，对`N`组数据分别估测`N`个$f ^\ast$.
* 每个$f^\ast$与$\hat f$之间存在误差，这个误差来自于%E[f\ast]%与$\hat f$的bias;
* 另外一个误差来自于$f^\ast$与$\overline f$的variance.

<img src="./bias&variance_of_estimator.jpg" width="500px">

**Note：**如何得到多个$f^\ast$呢？在不同的数据集上估计$f$。

举例：
分别用以下3种模型，学习100次，得到多个$f^\ast$，画图如下：
<img src="100f_star.jpg" width="500px">
从图中可以看到：
* 从上到下，模型的复杂程度越来越高；
* 简单的模型散步得很紧密，复杂的模型散布得比较开；
* 简单的模型受到数据(x)的影响较小（最极端的例子f(x)=c，最简单的模型，完全不受数据影响）。



# 总结