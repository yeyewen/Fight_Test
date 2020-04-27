# DP专题

## 13. 123. 买卖股票的最佳时机 III
给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。


>动态规划法
k表示售出股票的次数有三个状态
dp[i][k][0]表示第i个状态的没有持有股票的收益
dp[i][k][1]表示第i个状态的持有股票的收益
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n=len(prices)
        if n==0:
            return 0
        dp= [[[0 for i in range(2)] for i in range(3)] for i in range(n)]
        for k in range(3):
            dp[0][k][1] = -float('inf')
            dp[-1][k][1] = -float('inf')
        for i in range(n):
            for k in range(1,3):
                if i ==0:
                    dp[i][k][1]=-prices[i]
                dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
                dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k-1][0] - prices[i])
        return dp[n-1][2][0]
```


## 14.188. 买卖股票的最佳时机 IV
给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。

注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
这里需要注意的是当k笔交易则有k+1个状态

dp[i][k][0]表示第i个状态的没有持有股票的收益
dp[i][k][1]表示第i个状态的持有股票的收益
```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n=len(prices)
        if n==0:
            return 0
        ans=0
        if k > n/2:
            for i in range(1,n):
                if prices[i]>prices[i-1]:
                    ans+=  prices[i]-prices[i-1]
            return ans
        dp=[[[0 for i in range(2)] for i in range(k+1)] for i in range(n)]
        for i in range(n):
            for j in range(1,k+1):
                if i == 0 :
                    dp[i][j][1]=-prices[i]
                    continue
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i])
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j-1][0] - prices[i])
        return dp[n-1][k][0]
```
##15.309. 最佳买卖股票时机含冷冻期
增加一个状态

dp[i][k][0]表示第i个状态的没有持有股票的收益
dp[i][k][1]表示第i个状态的持有股票的收益
dp[i][k][2]表示第i个状态的没有持有股票的冷冻期的收益
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n=len(prices)
        if n==0:
            return 0
        dp=[[0 for i in range(3)] for i in range(n)]
        dp[0][1]=-prices[0]
        dp[0][2]=0
        for i in range(1,n):
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i])
            dp[i][2]=dp[i-1][0]
            dp[i][1]=max(dp[i-1][1],dp[i-1][2]-prices[i])
        return dp[n-1][0]

```
利用python多重复制进行状态压缩
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n=len(prices)
        if n==0:
            return 0
        dp0=0
        dp1=0
        dp2=-prices[0]
        for i in range(1,n):
            dp0,dp1,dp2=max(dp0,dp2+prices[i]),dp0,max(dp2,dp1-prices[i])
        return dp0

```


