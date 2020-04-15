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

## 4.[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)



### 4.1解法

>暴力法 有一个案例超时

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n=len(nums)
        if n==0:
            return 0
        ans=nums[0]
        for i in range(n):
            sum = 0
            for j in range(i,n):
                sum +=nums[j]
                ans =max(sum,ans)
        return ans
```

>动态规划状态转移方程

- 状态
  - dp[i]定义为数组nums中以num[i] 结尾的最大连续子串和
- 状态转移方程
  - dp[i] = max(dp[i-1]+nums[i],nums[i])
  
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n=len(nums)
        if n==1:
            return nums[0]
        dp=[nums[0]]*n
        ans=dp[0]
        for i in range(1,n):
            dp[i]=max(dp[i-1]+nums[i],nums[i])
            #for j in range(i,n):
            #    sum +=nums[j]
            ans =max(dp[i],ans)
        return ans
```
> 进行轻优化
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n=len(nums)
        if n==1:
            return nums[0]
        dp=nums[0]
        ans=dp
        for i in range(1,n):
            dp=max(dp+nums[i],nums[i])
            #for j in range(i,n):
            #    sum +=nums[j]
            ans =max(dp,ans)
        return ans
```
> 贪心法
使用单个数组作为输入来查找最大（或最小）元素（或总和）的问题，贪心算法是可以在线性时间解决的方法之一。
每一步都选择最佳方案，到最后就是全局最优的方案。

算法：
该算法通用且简单：遍历数组并在每个步骤中更新：

当前元素
当前元素位置的最大和（必须加到当前元素的最大和）
迄今为止的最大和

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n=len(nums)
        if n==1:
            return nums[0]
        now_sum=nums[0]
        max_sum=nums[0]
        for i in range(1,n):
            now_sum =max(nums[i],now_sum+nums[i])
            max_sum=max(now_sum,max_sum)
        return max_sum
```
