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

