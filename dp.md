# DP专题

## 3.[120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

### 3.1 题目

给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

例如，给定三角形：

[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。

### 3.2 解法



想到可以用递归

> 解法1：递归法
递归法python实现，最后一个数据会超时
```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if len(triangle)==0:
            return 0
        return self.dfs(triangle,0,0)
    def dfs(self,triangle,i,j):
        if i==len(triangle)-1:
            return triangle[i][j]
        left=triangle[i][j]+self.dfs(triangle,i+1,j)
        right=triangle[i][j]+self.dfs(triangle,i+1,j+1)
        return min(left,right)

```



> 解法2 自底向上，二维dp

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        #dp=triangle
        for i in range(len(triangle)-2,-1,-1):
            for j in range(len(triangle[i])):
                triangle[i][j]=triangle[i][j]+min(triangle[i+1][j],triangle[i+1][j+1])
        return triangle[0][0]
```
>两维变一维

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp=[0]*(len(triangle[-1])+1)
        for i in range(len(triangle)-1,-1,-1):
            for j in range(len(triangle[i])):
                dp[j]=triangle[i][j]+min(dp[j],dp[j+1])
        return dp[0]
```