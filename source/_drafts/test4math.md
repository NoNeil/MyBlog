---
title: test4math
categories: 
- Hexo
tags: 
- Hexo
- NexT
- mathjax
---

我安装主题是Next 5.1.4，已经能内嵌了mathjax，所以不需要再安装hexo-math插件。

NexT 5.1.4中默认mathjax是禁用, 启用mathjax的方式是，编辑主题的_config.yml，找到mathjax字段，改成:
```java
enable: true
```

Inline:

Simple inline $a = b + c$.

Block:

$$\frac{\partial u}{\partial t}
= h^2 \left( \frac{\partial^2 u}{\partial x^2} +
\frac{\partial^2 u}{\partial y^2} +
\frac{\partial^2 u}{\partial z^2}\right)$$


