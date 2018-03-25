---
title: git命令手册
date: 2018-03-24 14:14:09
updated: 2018-03-24 14:14:09
categories: git
tags: git
---

**【本博客随时更新】**

# 删除文件
同时删除远程Git仓库中的文件和本地文件系统中的文件：
```shell
$ git rm file1.txt
$ git commit -m "remove file1.txt"
$ git push origin branch_name 
```

如果只是想删除Git仓库里的文件，而不是本地文件系统里的文件，使用如下命令：
```shell
$ git rm --cached file1.txt
$ git commit -m "remove file1.txt"
$ git push origin branch_name 
```

# 修改commit message
```shell
$ git commit --amend -m "New commit message"
```