## 12.121. 买卖股票的最佳时机 II

### 12.1解法
>记录每次在当前状态下的每一个最低股票价格，再记录当前最低价格下的最优收益。
```python

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans=0
        for i in range(1,len(prices)):
            if prices[i]>prices[i-1]:
                ans+=  prices[i]-prices[i-1]
        return ans
```