---
title: Titanic Data Science Solution(翻译)
categories: kaggle
tags:
  - Machine Learning
  - kaggle
  - Python
date: 2018-03-18 18:55:25
updated: 2018-03-18 18:55:25
---

原文链接：[Titanic Data Science Solution](https://www.kaggle.com/startupsci/titanic-data-science-solutions)

# 工作流
kaggle比赛工作流包含7个阶段：
1. 理解问题；
2. 获取训练数据和测试数据；
3. 数据清理；
4. 分析、确定特征；
5. 建模、训练、预测；
6. 可视化、报告、提出解决问题的步骤和最终的方案；
7. 提交结果。

## 理解问题
仔细审题，理解是分类问题还是回归问题，或者其它。

## 获取数据
```python
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
```

## 数据清理
pandas包含一些获取数据描述的方法。

**数据有哪些特征？**
```python
print(train_df.columns.values)
```
> ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 
> 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']

预览数据：
```python
train_df.head()
```

**哪些特征是分类的？**
有些特征能将数据集分成多个子集。例如性别。可以对这些特征进行可视化，分析数据的分布。
* 分类的：Survived（是否幸存），Sex（性别），Embarked（登船口）
* 序列的：Pclass（舱位等级）

**哪些特征是数值的？**
这些数值特征是离散的、连续的、还是时间序列的？
* 连续的：Age（年龄），Fare（票价）
* 离散的：SibSp，Parch

**哪些特征的数据类型是混乱的？**
有些特征的数据类型既有数字的，也有字母的，这些特征在数据清理环节需要被处理。
* Ticket：数字和字母混合的
* Cabin（船舱）：字母的

**哪些特征包含错误数据？**
对于大型数据集来说比较困难，但是从较小的数据集中查看一些示例可能得出哪些特性需要改正。
* “Name”这个特征可能包含错误数据，因为有很多种方式来描述一个人的名字，如简称，名字字符串也可能附有圆括号或引号

**哪些特征包含空值、null、NaN等？**
包含空值的特征，是具体情况采用不同的方式进行填补。
* 这三个特征包含空值：Cabin > Age > Embarked
* Carbin和Age在测试集中不完整

**各个特征的数据类型是什么？**
```python
train_df.info()
print("_" * 50)
test_df.info()
```
Output：
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
________________________________________
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
```

**数值特征的分布是什么？**
这一步在早期分析中，有助于我们充分理解数据。
* 样本总数为891，占实际所有人数2224的40%
* “Survived”是一种具有0或1值的分类特征
* 样本中的幸存率大约是38%，实际的幸存率为32%
* 大多数乘客（超过75%）没有和父母或孩子一起旅行
* 近三成的乘客有兄弟姐妹 和/或 配偶
* 票价差异很大，很少有乘客（<1%）支付高达512美元的费用
* 年龄在65-80岁之间的老人很少（<1%）

```python
train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` 
# knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
```
Output:

||PassengerId|	Survived|	Pclass|	Age|	SibSp|	Parch|	Fare|
|-|-|-|-|-|
|count|	891.000000|	891.000000|	891.000000|	714.000000|	891.000000|	891.000000|	891.000000|
|mean|	446.000000|	0.383838|	2.308642|	29.699118|	0.523008|	0.381594|	32.204208|
|std|	257.353842|	0.486592|	0.836071|	14.526497|	1.102743|	0.806057|	49.693429|
|min|	1.000000|	0.000000|	1.000000|	0.420000|	0.000000|	0.000000|	0.000000|
|25%|	223.500000|	0.000000|	2.000000|	20.125000|	0.000000|	0.000000|	7.910400|
|50%|	446.000000|	0.000000|	3.000000|	28.000000|	0.000000|	0.000000|	14.454200
|75%|	668.500000|	1.000000|	3.000000|	38.000000|	1.000000|	0.000000|	31.000000|
|max|	891.000000|	1.000000|	3.000000|	80.000000|	8.000000|	6.000000|	512.329200|

**分类特征的分布是什么样的？**
* Name是唯一的（下表中，name的count为891，与样本总数一致）
* Sex的取值只有两种，其中male占多数，占比577/891=64.9%
* Cabin(船舱)有重复的，其中204个样本有船舱号，不同的船舱号有104个。所以存在多个样本的船舱号一样的情况。
**译者注：同一个船舱号中的人可能都幸存。由此甚至可以推出，相同姓氏的人可能都幸存**
* Embarked（登船口）有三种取值，其中从S口登船的人最多，有664个
* Ticket，有(891-681)/891=22%的样本的Tickt信息重复。
**译者注：可能是登记错误导致的数据错误**

```python
train_df.describe(include=['O'])
```

Output:

||	Name|	Sex|	Ticket|	Cabin|	Embarked|
|-|-|-|-|-|-|
|count|	    891|	891|	891|	204|	889|
|unique|	891|	2|	    681|	147|	3|
|top|	Lester, Mr. James|	male|	347082|	G6|	S|
|freq|	    1|	    577|	7|	4|	644|

## 基于数据分析进行假设
基于上述简单的分析得到一些假设，然后对数据进行深入的分析，进行验证。

**Correlating（寻找特征的关联性）**
* 我们想知道每个特征与结果的关系。我们希望在项目的早期就这样做，并将这些快速的相关性与项目后面的建模相关性进行匹配。

**Completing（将缺失数据的特征进行补全）**
* 我们可能想要完整的"Age"特征，因为它肯定与生存相关。
* 我们可能想要将“Embarked（登船口）”补全，因为它也可能与生存或另一个重要的特征相关。

**Correcting（纠正数据）**
* 在我们的分析中，可能要扔掉“Ticket”特征，因为它包含了高比率的重复(22%)，并且Ticket很可能与Survived无关
* 在训练和测试数据集中，Carbin（舱室）特征可能会被删除，因为它高度不完整或包含许多空值。
**译者注：训练模型的时候可以把Carbin特征扔掉，后期模型融合的时候这个特征还是可以用的**
* “PassengerId”特征可以删除，因为它对生存没有帮助。
* “Name”特征是相对不标准的，可能不会直接导致生存，所以可能会扔掉。

**Creating（创造特征）**
* 我们可能要基于Parch和SibSp创建一个新的特征，叫做“Family”，的家庭，以获得家庭成员的总数。
* 我们可能要从“Name”特征中提取“Title”作为一个新特征。
* 我们可能要为年龄层创造新的特征。这将一个连续的数字特征变成一个有序分类的特征。
**译者注：根据年龄建立直方图，每10岁为一个bin**
* 我们可能还想创建一个Fare range的特征。
**译者注：与Age特征类似**

**Classifying（分类）**
我们还可以根据前面提到的问题描述增加我们的假设。

* 女性(性=女性)更可能存活。
* 儿童(Age小于多少)更有可能存活。
* 舱位等级越高的乘客(Pclass=1)更有可能幸存下来。


未完待续...
