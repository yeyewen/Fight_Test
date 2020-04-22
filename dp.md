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
## 6.  887. 鸡蛋掉落 https://leetcode-cn.com/problems/super-egg-drop/

### 6.1 题目

你将获得 K 个鸡蛋，并可以使用一栋从 1 到 N  共有 N 层楼的建筑。

每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。

你知道存在楼层 F ，满足 0 <= F <= N 任何从高于 F 的楼层落下的鸡蛋都会碎，从 F 楼层或比它低的楼层落下的鸡蛋都不会破。

每次移动，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 X 扔下（满足 1 <= X <= N）。

你的目标是确切地知道 F 的值是多少。

无论 F 的初始值如何，你确定 F 的值的最小移动次数是多少？


### 6.2 解法

我们可以考虑使用动态规划来做这道题，状态可以表示成 (K,N)(K, N)(K,N)，其中 KKK 为鸡蛋数，NNN 为楼层数。当我们从第 XXX 楼扔鸡蛋的时候：


如果鸡蛋不碎，那么状态变成 (K,N−X)(K, N-X)(K,N−X)，即我们鸡蛋的数目不变，但答案只可能在上方的 N−X层楼了。也就是说，我们把原问题缩小成了一个规模为 (K,N−X)(K, N-X)(K,N−X) 的子问题；


如果鸡蛋碎了，那么状态变成 (K−1,X−1)(K-1, X-1)(K−1,X−1)，即我们少了一个鸡蛋，但我们知道答案只可能在第 XXX 楼下方的 X−1X-1X−1 层楼中了。也就是说，我们把原问题缩小成了一个规模为 (K−1,X−1)(K-1, X-1)(K−1,X−1) 的子问题。


这样一来，我们定义 dp(K,N)dp(K, N)dp(K,N) 为在状态 (K,N)(K, N)(K,N) 下最少需要的步数。根据以上分析我们可以列出状态转移方程：
dp(K,N)=1+min⁡1≤X≤N(max⁡(dp(K−1,X−1),dp(K,N−X)))dp(K, N) = 1 + \min\limits_{1 \leq X \leq N} \Big( \max(dp(K-1, X-1), dp(K, N-X)) \Big)
dp(K,N)=1+1≤X≤Nmin​(max(dp(K−1,X−1),dp(K,N−X)))
这个状态转移方程是如何得来的呢？对于 dp(K,N)dp(K, N)dp(K,N) 而言，我们像上面分析的那样，枚举第一个鸡蛋扔在的楼层数 XXX。由于我们并不知道真正的 FFF 值，因此我们必须保证 鸡蛋碎了之后接下来需要的步数 和 鸡蛋没碎之后接下来需要的步数 二者的 最大值 最小，这样就保证了在 最坏情况下（也就是无论 FFF 的值如何） dp(K,N)dp(K, N)dp(K,N) 的值最小。如果能理解这一点，也就能理解上面的状态转移方程，即最小化 max⁡(dp(K−1,X−1),dp(K,N−X))\max(dp(K-1, X-1), dp(K, N-X))max(dp(K−1,X−1),dp(K,N−X))。


```python
class Solution:
    def superEggDrop(self, K: int, N: int) -> int:
        memo = {}
        def dp(k, n):
            if (k, n) not in memo:
                if n == 0:
                    ans = 0
                elif k == 1:
                    ans = n
                else:
                    lo, hi = 1, n
                    while lo + 1 < hi:
                        x = (lo + hi) // 2
                        t1 = dp(k-1, x-1)
                        t2 = dp(k, n-x)
                        if t1 < t2:
                            lo = x
                        elif t1 > t2:
                            hi = x
                        else:
                            lo = hi = x
                    ans = 1 + min(max(dp(k-1, x-1), dp(k, n-x)) for x in (lo, hi))
                memo[k, n] = ans
            return memo[k, n]
        return dp(K, N)
```
## 7.354. 俄罗斯套娃信封问题

给定一些标记了宽度和高度的信封，宽度和高度以整数对形式 (w, h) 出现。当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算最多能有多少个信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

说明:
不允许旋转信封。

示例:

输入: envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出: 3 
解释: 最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。


### 7.1解法

>
动态规划代码，和别人同样的逻辑，但是python会超时（只会python，我可太惨了）
```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort()

        if len(envelopes)==0:
            return 0
        dp=[1]*len(envelopes)
        ans=1
        for i in range(1,len(envelopes)):
            for j in range(i):
                if envelopes[i][1]>envelopes[j][1] and envelopes[i][0]>envelopes[j][0]:
                    dp[i]=max(dp[i],dp[j]+1)
            ans=max(ans,dp[i])
        return ans
```

>二分查找
```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # [2, 3] [5, 4], [6, 4], [6, 7]
        # [3, 4, 4, 7, 1, 5, 6, 2, 3, 4, 5] => LIS
        # [3, 4, 5, 6]

        # Time complexity  : O(NlogN)
        # Space complexity : O(N)
        if envelopes == []: return 0
        envelopes.sort(key = lambda x: (x[0], -x[1]))
        tail = []
        res = 0
        for a, b in envelopes:
            index = bisect.bisect_left(tail, b)
            if index == len(tail):
                tail.append(b)
            else:
                tail[index] = min(tail[index], b)
            res = max(res, len(tail))
        return res


```
## 8.198.打家劫舍

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

示例 1:

输入: [1,2,3,1]
输出: 4
解释: 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。




### 8.1解法
状态转移方程

D[j]=max(D[j-1],D[j-2]+nums[j])

>
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        last,now=0,0
        for num in nums:
            last,now=now,max(last+num,now)
        return now
```

## 9.213. 打家劫舍 II （还有337打家劫舍III,用深度dfs递归解法）

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都围成一圈，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

示例 1:

输入: [2,3,2]
输出: 3
解释: 你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。

### 9.1解法
状态转移方程
延续198题，分成两个子问题，偷第一家或者不偷第一家
D[j]=max(D[j-1],D[j-2]+nums[j])

>
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) ==1:
            return nums[0]
        elif len(nums) ==0:
            return 0
        else:
            last1,last2,now1,now2=0,0,0,0
            for num in nums[1:]:
                last1,now1=now1,max(last1+num,now1)
            for num in nums[:-1]:
                last2,now2=now2,max(last2+num,now2)
            return max(now1,now2)
```

## 10. 337. 打家劫舍 III 

难度中等325收藏分享切换为英文关注反馈在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

示例 1:

输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.

示例 2:

输入: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1

输出: 9
解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.

### 10.1解法


不偷当前节点，两个儿子节点都要给出最多的钱
偷当前节点，则不能偷其两个儿子节点


>
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rob(self, root: TreeNode) -> int:
        #思路：递归
        def dp(node):
            #返回值是偷取该节点和不偷取该节点的最大收益
            if node == None:
                return (0,0)
            steal_left,not_steal_left = dp(node.left)
            steal_right,not_steal_right = dp(node.right)
            steal = not_steal_left + not_steal_right + node.val
            not_steal = max(steal_left,not_steal_left) + max(steal_right,not_steal_right)
            return (steal,not_steal)
        steal,not_steal = dp(root)
        return max(steal,not_steal)
```

