---
title: 553. Optimal Division
categories: LeetCode
tags: LeetCode
date: 2018-03-15 19:34:53
---
[LeetCode: 553. Optimal Division](https://leetcode.com/problems/optimal-division/)
# 问题描述
Given a list of **positive integers**, the adjacent integers will perform the float division. For example, [2,3,4] -> 2 / 3 / 4.

However, you can add any number of parenthesis at any position to change the priority of operations. You should find out how to add parenthesis to get the maximum result, and return the corresponding expression in string format. **Your expression should NOT contain redundant parenthesis.**

**Example:**
> Input: [1000,100,10,2]
> Output: "1000/(100/10/2)"
> Explanation:
> 1000/(100/10/2) = 1000/((100/10)/2) = 200
> However, the bold parenthesis in "1000/((100/10)/2)" are redundant, 
> since they don't influence the operation priority. So you should return > "1000/(100/10/2)". 
> 
> Other cases:
> 1000/(100/10)/2 = 50
> 1000/(100/(10/2)) = 50
> 1000/100/10/2 = 0.5
> 1000/100/(10/2) = 2

**Note**:

1. The length of the input array is [1, 10].
1. Elements in the given array will be in range [2, 1000].
1. There is only one optimal division for each test case.

# 分析
用**动态规划**求解

# 代码
1. 只求最大值，不用得到表达式
```Python
class Solution(object):
    def __init__(self):
        self.mat = []
        
    def optimalDivision(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        n, mat = len(nums), []
        for i in range(n):
            mat.append([0] * n)
        for margin in range(1, n):
            for i in range(n - margin):
                if margin == 1:
                    mat[i][i+margin] = nums[i] / nums[i+margin]  #右上角存最大值
                    mat[i+margin][i] = nums[i] / nums[i+margin]  #左下角存最小值
                else:
                    mat[i][i+margin] = max(nums[i] / mat[i+margin][i+1], mat[i][i+margin-1] / nums[i+margin])
                    mat[i+margin][i] = min(nums[i] / mat[i+1][i+margin], mat[i+margin-1][i] / nums[i+margin])

        return mat[0][n-1]
```

2. 用分治，求得表达式
```Python
class Solution(object):
    def __init__(self):
        self.mat = []
        self.exp = []
        
    def optimalDivision(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        
        def dev(nums, start, end):
            #print nums, 'start:', start, 'end:', end
            maxDev, minDev, expMax, expMin = 0, 10e8, '', ''
            if start == end:
                #print 'minDev:', nums[start], 'maxDev:', nums[start], 'expMin:', nums[start], 'expMax:', nums[start]
                self.mat[start][end], self.mat[end][start] = nums[start], nums[start]
                self.exp[start][end], self.exp[end][start] = str(nums[start]), str(nums[start])
                maxDev, minDev, expMax, expMin = nums[start], nums[start], str(nums[start]), str(nums[start])
            elif start + 1  == end:
                valDev = float(nums[start]) / nums[end]
                expDev = str(nums[start]) + "/" + str(nums[end])
                self.mat[start][end] = valDev  #右上角存最大值
                self.mat[end][start] = valDev  #左下角存最小值
                self.exp[start][end] = expDev  #右上角存最大值
                self.exp[end][start] = expDev  #左下角存最小值
                #print 'minDev:', valDev, 'maxDev:', valDev, 'expMin:', expDev, 'expMax:', expDev
                maxDev, minDev, expMax, expMin = valDev, valDev, expDev, expDev
            else:
                for split in range(start, end):
                    if self.mat[start][split] == 0:
                        left = dev(nums, start, split)
                    else:
                        left = self.mat[start][split], self.mat[split][start], self.exp[start][split], self.exp[split][start]
                    if self.mat[split+1][end] == 0: 
                        right = dev(nums, split+1, end)
                    else:
                        right = self.mat[split+1][end], self.mat[end][split+1], self.exp[split+1][end], self.exp[end][split+1]
                    
                    valMin, valMax = float(left[1]) / right[0], float(left[0]) / right[1]
                    if valMin < minDev:
                        minDev = valMin
                        if "/" in right[2]:
                            expMin = left[3] + '/(' + right[2] + ')'
                        else:
                            expMin = left[3] + '/' + right[2]
                    if valMax > maxDev:
                        maxDev = valMax
                        if "/" in right[3]:
                            expMax = left[2] + '/(' + right[3] + ')'
                        else:
                            expMin = left[2] + '/' + right[3]
                    
                self.mat[start][end], self.mat[end][start] = maxDev, minDev
                self.exp[start][end], self.exp[end][start] = expMax, expMin
                #print 'minDev:', minDev, 'maxDev:', maxDev, 'expMin:', expMin, 'expMax:', expMax
            return maxDev, minDev, expMax, expMin
        
        n = len(nums)
        for i in range(n):
            self.mat.append([0] * n)
            self.exp.append([''] * n)
            
        result = dev(nums, 0, n-1)
        return result[2]            
```
