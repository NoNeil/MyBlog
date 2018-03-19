---
title: 632.Smallest Range
categories: LeetCode
tags: LeetCode
date: 2018-03-15 17:56:02
---
[LeetCode: 632.Smallest Range](https://leetcode.com/problems/smallest-range/)
# 问题描述
You have `k` lists of sorted integers in ascending order. Find the **smallest** range that includes at least one number from each of the `k` lists.

We define the range `[a,b]` is smaller than range `[c,d]` if `b-a < d-c` or `a < c` if `b-a == d-c`.

**Example 1:**
> Input:[[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
> Output: [20,24]
> Explanation: 
> List 1: [4, 10, 15, 24,26], 24 is in range [20,24].
> List 2: [0, 9, 12, 20], 20 is in range [20,24].
> List 3: [5, 18, 22, 30], 22 is in range [20,24].

**Note:**
1. The given list may contain duplicates, so ascending order means >= here.
2. 1 <= k <= 3500
3. -105 <= value of elements <= 105.
4. For Java users, please note that the input type has been changed to List<List<Integer>>. And after you reset the code template, you'll see this point.


# 题意

给定k个数组，找出一个最小的**区间**，使得区间内包含每个数字内至少一个数。

# 分析

用一个优先队列，里面存k个分别来自k个数组的数。

每次从队列里弹出一个最小值，并从弹出值的数组里，添加下一个值。

每次弹出时，计算但是的range，如果比之前的小，就替换掉之前的range，作为一个新结果。

1. 队列，能满足区间里同时来自k个数组的k个数；
2. 最小区间，通过这个来满足：每次往里队列里添加的都是同budga下一个数（也就是紧接着最小的数），如果当期range小于上一个range就替换之。

# 代码
```Java
public class Solution {
    public int[] smallestRange(List<List<Integer>> nums) {
        PriorityQueue<Element> priorityQueue = new PriorityQueue<Element>(new Comparator<Element>() {
            @Override
            public int compare(Element o1, Element o2) {
                return o1.value - o2.value;
            }
        });
        int maxValue = Integer.MIN_VALUE;
        for (int i = 0; i < nums.size(); i++) {
            Element element = new Element(nums.get(i).get(0), 0, i);
            priorityQueue.offer(element);
            maxValue = Math.max(maxValue, nums.get(i).get(0));
        }
        int range = Integer.MAX_VALUE;
        int start = -1, end = -1;
        while(priorityQueue.size() == nums.size()){
            Element popElement = priorityQueue.poll();
            if (maxValue - popElement.value < range) {
                start = popElement.value;
                end = maxValue;
                range = maxValue - popElement.value;
            }
            int index = popElement.index + 1;
            if (index < nums.get(popElement.budge).size()){
                Element element = new Element(nums.get(popElement.budge).get(index), index, popElement.budge);
                priorityQueue.offer(element);
                maxValue = Math.max(maxValue, element.value);
            }
        }
        return new int[]{start, end};
    }
}

class Element{
    public int value;
    public int index;
    public int budge;

    public Element(int v, int i, int b){
        this.value = v;
        this.index = i;
        this.budge = b;
    }
}
```