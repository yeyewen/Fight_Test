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
## 15.309. 最佳买卖股票时机含冷冻期
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
利用python多重赋值进行状态压缩
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
## 16.714. 买卖股票的最佳时机含手续费
仔细分析问题，在进行收益计算时减去手续费即可
```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n=len(prices)
        if n==0:
            return 0
        dp=[[0 for i in range(2)] for i in range(n)]
        dp[0][1]=-prices[0]
        for i in range(1,n):
            dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i]-fee)
            dp[i][1]=max(dp[i-1][1],dp[i-1][0]-prices[i])
        return dp[n-1][0]


```
状态压缩
```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n=len(prices)
        if n==0:
            return 0
        dp0=0
        dp1=-prices[0]
        for i in range(1,n):
            dp0,dp1=max(dp0,dp1+prices[i]-fee),max(dp1,dp0-prices[i])
        return dp0


```
## 17. 72.编辑距离

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：


	插入一个字符
	删除一个字符
	替换一个字符

有两个单词，三种操作，就有六种操作方法。
但我们可以发现，如果我们有单词 A 和单词 B：对单词 A 删除一个字符和对单词 B 插入一个字符是等价的。
例如当单词 A 为 doge，单词 B 为 dog 时，我们既可以删除单词 A 的最后一个字符 e，得到相同的 dog，也可以在单词B末尾添加一个字符e，得到相同的doge；
同理，对单词 B 删除一个字符和对单词 A 插入一个字符也是等价的；
对单词 A 替换一个字符和对单词 B 替换一个字符是等价的。例如当单词 A 为 bat，单词B为cat时，我们修改单词A的第一个字母b->c，和修改单词B的第一个字母c->b是等价的

这样本质不同的操作实际上只有三种：
在单词A中插入一个字符
在单词B种插入一个字符
修改单词A中的一个字符
若A和B的最后一个字母相同：
D[i][j]=min(D[i][j−1]+1,D[i−1][j]+1,D[i−1][j−1])
若A和B的最后一个字母不同：
D[i][j]=min(D[i][j−1]+1,D[i−1][j]+1,D[i−1][j−1]+1)
```python
class Solution:
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n = len(word1)
        m = len(word2)
        
        # 有一个字符串为空串
        if n * m == 0:
            return n + m
        
        # DP 数组
        D = [ [0] * (m + 1) for _ in range(n + 1)]
        
        # 边界状态初始化
        for i in range(n + 1):
            D[i][0] = i
        for j in range(m + 1):
            D[0][j] = j
        
        # 计算所有 DP 值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = D[i - 1][j] + 1
                down = D[i][j - 1] + 1
                left_down = D[i - 1][j - 1] 
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                D[i][j] = min(left, down, left_down)
        
        return D[n][m]
``` 
## 44.通配符匹配
给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*'  的通配符匹配。
该题与编辑距离这道题比较像，用类似的思路建立状态转移方法
动态规划：dp[i][j]表示：s的前i个字符与p的前j个字符是否匹配
状态转移方程
如果s1的第 i 个字符和s2的第 j 个字符相同，或者s2的第 j 个字符为 “?”
f[i][j] = f[i - 1][j - 1]
如果s2的第 j 个字符为 *
若s2的第 j 个字符匹配空串, f[i][j] = f[i][j - 1]
若s2的第 j 个字符匹配s1的第 i 个字符, f[i][j] = f[i - 1][j]
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        s = '0'+s
        p = '0'+p
        # dp[i][j]表示：s的前i个字符与p的前j个字符是否匹配
        dp = [[False for _ in range(len(p))] for _ in range(len(s))]

        # 初始化
        dp[0][0] = True  # 空字符串与空字符串相匹配
        for i in range(1, len(p)):
            dp[0][i] = dp[0][i-1] and p[i] == '*'
        for i in range(1, len(s)):
            dp[i][0] = False
        
        # 动态规划
        for i in range(1, len(s)):
            for j in range(1, len(p)):
                if s[i] == p[j] or p[j] == '?':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j] == '*':
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
        return dp[-1][-1]
``` 

##  10. 正则表达式匹配

该题与通配符匹配题很像，用同样动态规划代码思路解题
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        s = '0'+s
        p = '0'+p
        # dp[i][j]表示：s的前i个字符与p的前j个字符是否匹配
        dp = [[False for _ in range(len(p))] for _ in range(len(s))]

        # 初始化
        dp[0][0] = True  # 空字符串与空字符串相匹配
        for i in range(1, len(p)):
            dp[0][i] = dp[0][i-2] and p[i] == '*'
        for i in range(1, len(s)):
            dp[i][0] = False
        
        # 动态规划
        for i in range(1, len(s)):
            for j in range(1, len(p)):
                if s[i] == p[j] or p[j] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j] == '*':
                    if p[j-1]==s[i] or p[j-1]==".":
                        dp[i][j] = dp[i-1][j] or dp[i][j-2]
                    else:
                        dp[i][j]=dp[i][j-2]
                else:
                    dp[i][j]=False
        return dp[-1][-1]
```
