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
### 5.1 题目

给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字）。

 

示例 1:

输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。


### 5.2 解法


> 解法1：暴力法 python实现，最后一个数据会超时，我再也不试暴力法了
```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n=len(nums)
        dp=nums[0]
        for i in range(n):
            multi=1
            for j in range(i,n):
                multi*=nums[j]
                dp=max(dp,multi)
        return dp
```



> 解法2 动态规划解法，需利用python多重赋值

我们先定义一个数组 dpMax，用 dpMax[i] 表示以第 i 个元素的结尾的子数组，乘积最大的值，也就是这个数组必须包含第 i 个元素。
那么 dpMax[i] 的话有几种取值。

当 nums[i] >= 0 并且dpMax[i-1] > 0，dpMax[i] = dpMax[i-1] * nums[i]
当 nums[i] >= 0 并且dpMax[i-1] < 0，此时如果和前边的数累乘的话，会变成负数，所以dpMax[i] = nums[i]
当 nums[i] < 0，此时如果前边累乘结果是一个很大的负数，和当前负数累乘的话就会变成一个更大的数。所以我们还需要一个数组 dpMin 来记录以第 i 个元素的结尾的子数组，乘积最小的值。

当dpMin[i-1] < 0，dpMax[i] = dpMin[i-1] * nums[i]
当dpMin[i-1] >= 0，dpMax[i] =  nums[i]



当然，上边引入了 dpMin 数组，怎么求 dpMin 其实和上边求 dpMax 的过程其实是一样的。
参考题解 https://leetcode-cn.com/problems/maximum-product-subarray/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by--36/

这里保存状态可以是一个两维，两个一维，利用python的多重赋值压缩为两个常数空间
```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n=len(nums)
        dp_min=nums[0]
        dp_max=dp_min
        ans=dp_min
        for i in range(1,n):
            if nums[i]>0:
                dp_max,dp_min=max(dp_max*nums[i],nums[i],dp_min*nums[i]),min(dp_min*nums[i],nums[i],dp_max*nums[i])
                #dp_min=min(dp_min*nums[i],nums[i],dp_max*nums[i])
            else :
                dp_max,dp_min=max(dp_min*nums[i],nums[i]),min(dp_max*nums[i],nums[i])
                #dp_min=min(dp_max*nums[i],nums[i])
            ans=max(dp_min,ans,dp_max)
        return ans
```
>解法二https://leetcode-cn.com/problems/maximum-product-subarray/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by--36/
这个题解写的很巧妙

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n=len(nums)
        dp1=1
        ans=nums[0]
        dp2=1
        for i in range(0,n):
            dp1 *=nums[i]
            ans=max(dp1,ans)
            if nums[i]==0:
                dp1=1
        for i in range(n-1,0,-1):
            dp2 *=nums[i]
            ans=max(dp2,ans)
            if nums[i]==0:
                dp2=1
        return ans
```
