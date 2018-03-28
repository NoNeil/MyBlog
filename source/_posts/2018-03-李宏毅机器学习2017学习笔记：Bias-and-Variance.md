---
title: 李宏毅机器学习2017学习笔记：Bias and Variance
description: 如何分析error来自哪里？是bias偏大，还是variance偏大？分别如何解决？最后，使用N折交叉验证来选择模型。
categories: Machine Learning
tags:
  - Machine Learning
  - 李宏毅
  - Bias
  - Variance
date: 2018-03-26 20:29:58
updated: 2018-03-26 20:29:58
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
* 简单的模型受到数据(x)的影响较小（最极端的例子$f(x)=c$，最简单的模型，完全不受数据影响）。

对5000个 $f^\ast$ 求平均，画出来的蓝色线如下图：
**比较bias：**
* 左边的模型比较简单，求平均之后离 $\hat f$ 较远，bias较大；
* 右边的模型比较复杂，求平均之后离 $\hat f$ 较近，bias较小。

**比较variance：**
* 左边的模型输出值均在 $\hat f$ 附近，variance较小；
* 右边的模型输出值散布较开，variance较大。
<img src="5000f_star.jpg" width="500px">

# 如何处理bias偏大的情况？
分析方法：
* 如果模型不能很好地拟合训练数据，说明bias很大。【欠拟合】
* 如果模型能拟合训练数据，但是在测试数据上误差较大，那么很可能variance很大。【过拟合】

如果是bais很大，那么需要使模型更加复杂：
* 加更多的特征
* 用更加复杂的模型

如果是variance很大，那么：
* 收集更大的模型。如下图，100个样本训练的模型比10个样本训练的模型散布更加紧凑。
* 正则化。如下图，从左往右，正则项系数逐渐增大。
<img src="large_variance.jpg" width="500px">


# 如何选择模型？【交叉验证】
作业提供了一个训练集和公开的测试集，提交作业的时候，需要在私有的测试集上对提交的结果进行测试。

如果在训练集上训练了3个模型，在测试集上，『Model 3』 的 Error=0.5 最小，于是将『Model 3』的结果提交上去，结果Error>0.5。怎么办？
<img src="./directly_select_model.jpg" width="500px">

**N折交叉验证**
使用『N折交叉验证』来选择最优的模型，过程如下。
1. 将训练集分成`N`份；
2. 对于每个模型而言，依次将第`i`份拿出来作验证，在其他的`N-1`数据上训练，得到`N`个训练误差。求平均，得到平均训练误差；
3. 对于多个训练模型而言，取平均训练误差最小的模型。
<img src="./n_fold_cv.jpg" width="500px">


# 总结
1. 如果在训练集上误差较大，那么bias较大，说明是欠拟合。考虑增加特征，或者换更复杂的模型；
2. 如果在训练集上误差较小，在测试集上误差较大，过拟合。考虑收集更多的数据，或者采用正则化。
3. 多个候选模型之间如何选择？采用N折交叉验证。