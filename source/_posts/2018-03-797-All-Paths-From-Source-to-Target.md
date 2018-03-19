---
title: 797.All Paths From Source to Target
categories: LeetCode
tags: LeetCode
date: 2018-03-16 09:59:15
comment: true
---

[LeetCode: 797.All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/)

# 问题描述
Given a directed, acyclic graph of `N` nodes.  Find all possible paths from node `0` to node `N-1`, and return them in any order.

The graph is given as follows:  the nodes are 0, 1, ..., graph.length - 1.  graph[i] is a list of all nodes j for which the edge (i, j) exists.

**Example:**
```
Input: [[1,2], [3], [3], []] 
Output: [[0,1,3],[0,2,3]] 
Explanation: The graph looks like this:
0--->1
|    |
v    v
2--->3
There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.
```
**Note:**

* The number of nodes in the graph will be in the range [2, 15].
* You can print different paths in any order, but you should keep the order of nodes inside one path.
 
# 分析
使用dfs

# 代码
```Java
class Solution {
    
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        int n = graph.length;   // 结点个数
        boolean[] visited = new boolean[n];  // 记录i结点是否被访问过
        List<Integer> path = new ArrayList<Integer>(); // 路径
        path.add(0);            // 初始化路径
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        
        dfs(graph, visited, path, 0, n - 1, result);
        return result;
    }
    
    /**
    * int[] visited 表示节点i是否被访问过
    * List<Integer> path 表示路径
    * int curr 表示当前访问的结点
    * int target 表示目标结点，也就是graph.length-1
    * List<List<Integer>> result 存储所有的满足条件的路径
    */
    public void dfs(int[][] graph, boolean[] visited, List<Integer> path, int curr, int target, List<List<Integer>> result){        
        //System.out.println("curr:" + curr);
        if (curr == target){
            result.add(new ArrayList<Integer>(path));
            return;
        }
        for(int i: graph[curr]){
            //System.out.println("curr:" + curr + ", visite:" + i + ", status:" + visited[i]);
            if (!visited[i]){   // 如果i未被访问
                visited[i] = true;  // 访问i节点，将i添加到path中
                path.add(i);
                dfs(graph, visited, path, i, target, result);
                visited[i] = false; // 不访问i节点，将i从path中删除
                path.remove(path.size() - 1);
            }            
        }
    }
}
```