# 剑指offer专题

## 面试题11. 旋转数组的最小数字

简单46把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。
例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。

二分查找
```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        i,j=0,len(numbers)-1
        while i<j:
            m=(i+j)//2
            if numbers[m]>numbers[j]:i=m+1
            elif numbers[m]<numbers[j]:j=m
            else:j-=1
        return numbers[i]
      
```

## 面试题18. 删除链表的节点

```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if not head:
            return head
        if head.val==val:
            return head.next
        cur=head
        while head and  head.next:
            if head.next.val==val:
                head.next=head.next.next
            head=head.next            
        return cur
```
## 面试题12. 矩阵中的路径

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。
如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。


## dfs专练
### 面试题 04.04. 检查平衡性
实现一个函数，检查二叉树是否平衡。在这个问题中，平衡树的定义如下：任意一个节点，其两棵子树的高度差不超过 1。
```python
class Solution:
    def Depth(self,root:TreeNode) -> bool:
        if root:
            return 1+max(self.Depth(root.left),self.Depth(root.right))
        return 0
    def isBalanced(self, root: TreeNode) -> bool:
        if not root:
            return True
        if abs(self.Depth(root.left)-self.Depth(root.right)) >1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)
	
```

### 面试题 04.05. 合法二叉搜索树 
二叉搜索树：
二叉搜索树的中序遍历必然是一个递增序列利用这一点
递归方法
```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        ans=[]
        def dfs(root):
            if root:
                dfs(root.left)
                ans.append(root.val)
                dfs(root.right)
        dfs(root) 
        return ans == sorted(set(ans))

```
非递归方法，使用辅助栈方法
```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack=[]
        p=root
        res=[]
        while p or stack:
            while P:
                stack.append(p)
                p=p.left
            if stack:
                node=stack.pop()
                res.append(node.val)
                p=node.right
        return res==sorted(set(res))

```

### 530. 二叉搜索树的最小绝对差
采用递归的方式中序遍历，得到一组遍历数组，再求最小值。代码如下：这样很慢

```python
class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        def preorder(root):
            if not root:
                return []
            else:
                return preorder(root.left)+ [root.val] + preorder(root.right)
        target=preorder(root)
        if len(target)<=1:
            return 
        min_=target[1]-target[0]
        for i in range(1,len(target)):
            min_=min(abs(target[i]-target[i-1]),min_)
        return min_
```

```python
class Solution(object):
    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        prev = None
        min_dif = float('inf')
        stack = []
        while stack != [] or root != None:
            while root:
                stack.append(root)
                root = root.left
            node = stack.pop()
            if prev != None:
                dif = node.val - prev.val
                if dif < min_dif:
                    min_dif = dif
            prev = node
            root = node.right
        return min_dif

```

### 538. 把二叉搜索树转化为累加树 
给定一个二叉搜索树（Binary Search Tree），把它转换成为累加树（Greater Tree)，使得每个节点的值是原来的节点值加上所有大于它的节点值之和。

本题是关于二叉搜索树的问题，那我们第一想到的就是中序遍历，这是二叉搜索树的一个非常重要的性质，二叉搜索树的中序遍历是一个递增的有序序列。
本道题我们需要将其转换为累加树，使得每个节点的值是原来的节点值加上所有大于它的节点值之和。那我们看下面的例子：

```python
class Solution(object):
    def __init__(self):
        self.total = 0

    def convertBST(self, root):
        if root is not None:
            self.convertBST(root.right)
            self.total += root.val
            root.val = self.total
            self.convertBST(root.left)
        return root

```

### 543. 二叉树的直径
难度简单339收藏分享切换为英文关注反馈给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

一开始以为是左子树与右子树的深度和
代码如下
```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        max_=0
        def Depth(root):
            if not root:
                return 0
            else:
                l=Depth(root.left)
                r=Depth(root.right)
                return 1+max(Depth(root.left),Depth(root.right))
        if not root:
            return 0
        else:
            return Depth(root.left)+Depth(root.right)

```
但是有样例不能通过，在看过https://leetcode-cn.com/problems/diameter-of-binary-tree/solution/hot-100-9er-cha-shu-de-zhi-jing-python3-di-gui-ye-/ 的题解以后发现问题


```python
class Solution:
    
    def __init__(self):
        self.max = 0
    
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.depth(root)
        
        return self.max
        
    def depth(self, root):
        if not root:
            return 0
        l = self.depth(root.left)
        r = self.depth(root.right)
        '''每个结点都要去判断左子树+右子树的高度是否大于self.max，更新最大值'''
        self.max = max(self.max, l+r)
        
        # 返回的是高度
        return max(l, r) + 1

```

### 572. 另一个树的子树
给定两个非空二叉树 s 和 t，检验 s 中是否包含和 t 具有相同结构和节点值的子树。s 的一个子树包括 s 的一个节点和这个节点的所有子孙。s 也可以看做它自身的一棵子树。

```python
class Solution:
    def check(self,p,q):
        if not p and not q:
            return True
        elif not p or not q:
            return False
        else:
            return self.check(p.left,q.left) and self.check(p.right,q.right) and p.val==q.val
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if not t:
            return True
        if not s:
            return False
        return self.check(s,t) or self.isSubtree(s.left,t) or self.isSubtree(s.right,t)

```

### 100.相同的树
给定两个二叉树，编写一个函数来检验它们是否相同。
如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

```python
class Solution:
    def check(self,p,q):
        if not p and not q:
            return True
        elif not p or not q:
            return False
        else:
            return self.check(p.left,q.left) and self.check(p.right,q.right) and p.val==q.val
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        if not p and q:
            return False
        if p and not q:
            return False
        return self.check(p,q) 
```
简化递归过程
```python
class Solution:

    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        if not p and q:
            return False
        if p and not q:
            return False
        if p.val!=q.val:
            return False
        return self.isSameTree(p.left,q.left) and self.isSameTree(q.right,p.right) 
```
### 102. 二叉树的层序遍历

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        helper=[root]
        res=[]
        while helper:
            tmp1=[]
            tmp2=[]
            for node in helper:
                tmp1.append(node.val)
                if node.left:
                    tmp2.append(node.left)
                if node.right:
                    tmp2.append(node.right)
            res.append(tmp1)
            helper=tmp2
        return res
```

### 面试题15. 二进制中1的个数
该题要注意负数左移和右移方法，
左移n位，最左边的n位将被丢弃，同时在最右边补上n个0
右移 无符号数值在左边补0，有符号数值在最左边补一
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count=0
        while n:
            #if n&flag:
            count+=1
            n=n-1&n
        return count
```
### 面试题22. 链表中倒数第k个节点
快慢指针法
```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        fast,low=head,head
        while k-1>0:
            fast=fast.next
            k-=1
        while fast.next:
            fast=fast.next
            low=low.next
        return low
```
### 面试题17. 打印从1到最大的n位数
```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        ans=[]
        for i in range(1,10**n):
            ans.append(i)
        return ans

```
### 面试题55 - I. 二叉树的深度
```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        l=self.maxDepth(root.left)
        r=self.maxDepth(root.right)
        return max(l,r)+1
```


### 面试题25. 合并两个排序的链表
```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not  l2:
            return l1
        l3=None
        if l1.val<l2.val:
            l3=l1
            l1.next=self.mergeTwoLists(l1.next,l2)
        else:
            l3=l2
            l2.next=self.mergeTwoLists(l1,l2.next)
        return l3
```

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        ans=cur=ListNode(None)
        while l1 and l2:
            if l1.val<l2.val:
                ans.next,l1=l1,l1.next
            else:
                ans.next,l2=l2,l2.next
            ans=ans.next
        ans.next=l1 if l1 else l2
        return cur.next
```
### 面试题54. 二叉搜索树的第k大节点
```python
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        a=[]
        def dfs(root):
            if root: 
                dfs(root.left)
                a.append(root.val)
                dfs(root.right)
        dfs(root)
        return a[-k]
```

中序遍历的逆序，提前截止
```python

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        def dfs(root):
            if not root: 
                return
            dfs(root.right)
            if self.k == 0: 
                return
            self.k -= 1
            if self.k == 0: 
                self.res = root.val
            dfs(root.left)

        self.k = k
        dfs(root)
        return self.res
```
### 用两个栈来实现队列
```python

class CQueue:

    def __init__(self):
        self.stack1,self.stack2=[],[]


    def appendTail(self, value: int) -> None:
        self.stack1.append(value)
        #self.stack2.append(value)
    def deleteHead(self) -> int:
        if self.stack2:
            return self.stack2.pop()
        if not self.stack1:
            return -1
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop()


```
### 面试题57 - II. 和为s的连续正数序列
暴力法超时
```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        limit=(target+1)//2
        ans=[]
        for i in range(1,limit):
            sum_=i
            a=[i]
            for j in range(i+1,limit+1):
                sum_ +=j
                a.append(j)
                if sum_==target:
                    #sum_=0
                    ans.append(a)
                    break
                    #sum_=0
        return ans
```
### 面试题68 - I. 二叉搜索树的最近公共祖先

dfs
```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root:
            if root.val>p.val and root.val>q.val:
                return self.lowestCommonAncestor(root.left,p,q)
            elif root.val<p.val and root.val<q.val:
                return self.lowestCommonAncestor(root.right,p,q)
            else:
                return root
        
```
### 面试题03. 数组中重复的数字
```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        if len(nums)==0:
            return 
        nums.sort()
        for i in range(len(nums)-1):
            if nums[i]==nums[i+1]:
                return nums[i]
        
```
利用集合实现更高效
```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        if len(nums)==0:
            return 
        #nums.sort()
        s=set([])
        for i in nums:
            if i not in s:
                s.add(i)
            else:
                return i

```
### 面试题39. 数组中出现次数超过一半的数字
使用哈希表进行统计
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        d={}
        for i in nums:
            d[i] = d.get(i,0)+1
        for i in d:
            if d[i]>len(nums)//2:
                return i
```
使用python内置函数
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        s=set(nums)
        for i in s:
            if nums.count(i) >len(nums)/2:
                return i 
            
```
方法二：排序找中点
方法三: 摩尔投票法（待完善）
### 面试题57. 和为s的两个数字
输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
哈希表解法
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap={}
        for idx, num in enumerate(nums):
            if target - num in hashmap:
                return [num,target-num]
            else:
                hashmap[num] = idx
```
双指针法
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i,j=0,len(nums)-1
        while i<j:
            sum_=nums[i]+nums[j]
            if sum_<target:
                i +=1
            elif sum_>target:
                j-=1
            else:
                return [nums[i],nums[j]]
```
### 面试题52. 两个链表的第一个公共节点
剑指offer上的解法
```python
class Solution:
    def getLen(self,root):
        if not root:
            return 0
        return self.getLen(root.next)+1
    def same(self,p,q):
        if p and q:
            if p!=q:
                return self.same(p.next,q.next)
            else:
                return p
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        A_len=self.getLen(headA)
        B_len=self.getLen(headB)
        if A_len <=B_len:
            dis=B_len-A_len
            while dis:
                headB=headB.next
                dis-=1
            return self.same(headA,headB)
        else:
            dis=A_len-B_len
            while dis:
                headA=headA.next
                dis-=1
            return self.same(headA,headB)         
```
双指针相遇解法
```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        node1, node2 = headA, headB
        
        while node1 != node2:
            node1 = node1.next if node1 else headB
            node2 = node2.next if node2 else headA

        return node1
       
```
### 面试题21 调整数组顺序使奇数位于偶数前面
双指针法、注意循环的终止条件
```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        i,j=0,len(nums)-1
        while i<j:
            while i<j and nums[i]%2!=0:
                i+=1
            while i<j and nums[j]%2==0:
                j -=1
            nums[i],nums[j]=nums[j],nums[i]

        return nums
```
python暴力遍历法
```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        return [n for n in nums if n%2==1] + [n for n in nums if n%2==0]

```

### 面试题62 圆圈中最后剩下的数字
真暴力模拟
```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        i, a = 0, list(range(n))
        while len(a) > 1:
            i = (i + m - 1) % len(a)
            a.pop(i)
        return a[0]

```
还有递归和迭代的方法

### 面试题42. 连续子数组的最大和
动态规划法为本题时间复杂度和空间复杂度最优
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n=len(nums)
        dp=[nums[0]]*n
        ans=nums[0]
        for i in range(1,n):
            dp[i]=max(dp[i-1]+nums[i],nums[i])
            ans=max(ans,dp[i])
        return ans
```
### 面试题50. 第一个只出现一次的字符
```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        d={}
        for i in s:
            d[i]=d.get(i,0)+1
        for i in d:
            if d[i]==1:
                return i
        return " "
```
哈希表解法
第二种哈希构造方法更快一些
```python
class Solution:
    def firstUniqChar(self, s: str) -> str:
        dic = {}
        for c in s:
            dic[c] = not c in dic
        for c in s:
            if dic[c]: return c
        return ' '
```
思考题 ：为什么Python 3.6以后字典有序并且效率更高？
### 面试题60.n个骰子的点数
n=1时候的结果不用多说了
n=2时， 将第二个骰子(六个1/6)添加到第一个骰子的结果(六个1/6)上去，得到了n=2的结果
n=3时， 将第三个骰子(六个1/6)添加到第二个骰子的结果(n=2的结果在上个循环已经求得)上去
n=4时， 将第四个骰子(六个1/6)添加到第三个骰子的结果(n=3的结果在上个循环已经求得)上去
n=5,6,7....
以此类推
循环结束就得到了答案。
```python
class Solution:
    def twoSum(self, n: int) -> List[float]:
        dp=[1/6]*6
        for i in range(1,n):
            tmp=[0]*(5*i+6)
            for j in range(len(dp)):
                for k in range(6):
                    tmp[j+k]+= dp[j]*1/6 
            dp=tmp
        return dp

```
使用哈希优化
```python
class Solution:
    def twoSum(self, n: int) -> List[float]:
        all_num = 6 ** n
        old,new = {0:1},{}
        for _ in range(n):
            for i in range(1,7):
                for n in old:
                    new[n + i] = new.get(n + i, 0) + old.get(n)
            old = new
            new = {}
        return [old[i]/all_num for i in sorted(old)]
```
### 面试题53 - I. 在排序数组中查找数字 I

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums)==0:
            return 0
        if len(nums)==1:
            if nums[0] ==target:
                return 1
            else:
                return 0
        i,j=0,len(nums)-1
        while i<j:
            if nums[i]< target:
                i+=1
            if nums[j]>target:
                j-=1
            if i==j and nums[i]!=nums[j]:
                return 0
            if nums[i]==target and nums[j]==target:
                return j-i+1
        return 0
```
```python
class Solution:
    def search(self, nums: [int], target: int) -> int:
        # 搜索右边界 right
        i, j = 0, len(nums) - 1
        while i <= j:
            m = (i + j) // 2
            if nums[m] <= target: i = m + 1
            else: j = m - 1
        right = i
        # 若数组中无 target ，则提前返回
        if j >= 0 and nums[j] != target: return 0
        # 搜索左边界 left
        i = 0
        while i <= j:
            m = (i + j) // 2
            if nums[m] < target: i = m + 1
            else: j = m - 1
        left = j
        return right - left - 1
```
### 面试题61. 扑克牌中的顺子
```python
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        repeat = set()
        ma, mi = 0, 14
        for num in nums:
            if num == 0: 
                continue # 跳过大小王
            ma = max(ma, num) # 最大牌
            mi = min(mi, num) # 最小牌
            if num in repeat:
                 return False # 若有重复，提前返回 false
            repeat.add(num) # 添加牌至 Set
        return ma - mi < 5 # 最大牌 - 最小牌 < 5 则可构成顺子 
```
### 面试题58 - I. 翻转单词顺序
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s.strip()
        s=s.split()
        return ' '.join(s[::-1],)
```
### 面试题67. 把字符串转换成整数
```python
class Solution:
    def strToInt(self, str: str) -> int:
        str = str.strip() # 删除首尾空格
        if not str: return 0 # 字符串为空则直接返回
        res, i, sign = 0, 1, 1
        int_max, int_min, bndry = 2 ** 31 - 1, -2 ** 31, 2 ** 31 // 10
        if str[0] == '-': sign = -1 # 保存负号
        elif str[0] != '+': i = 0 # 若无符号位，则需从 i = 0 开始数字拼接
        for c in str[i:]:
            if not '0' <= c <= '9' : break # 遇到非数字的字符则跳出
            if res > bndry or res == bndry and c > '7': return int_max if sign == 1 else int_min # 数字越界处理
            res = 10 * res + ord(c) - ord('0') # 数字拼接
        return sign * res
```
### 面试题16. 数值的整数次方
硬做超时
时间复杂度为O（n）
```python
class Solution:
    def pow_(self,x,n):
        ans=1
        for _ in range(n):
            ans *=x
        return ans
    def myPow(self, x: float, n: int) -> float:
        if n==0:
            return 1
        if n >0:
            return self.pow_(x,n)
        if n<0:
            return self.pow_(1/x,-n)
```
快速幂方法O（logn） 二分法
时间复杂度为
```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0: return 0
        res = 1
        if n < 0: x, n = 1 / x, -n
        while n:
            if n & 1: res *= x
            x *= x
            n >>= 1
        return res
```
### 面试题20. 表示数值的字符串

```python
class Solution:
    def isNumber(self, s: str) -> bool:
        try:float(s);return True
        except:return False
```
```python
class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if '' == s[0]:
            return False
        else:
            s1 = s.rstrip().strip()
            if " " in s1:
                return False
            # 先对+ -号处理
            for i in ['+','-']:
                if i in s1:
                    if s1.rindex(i) == 0:
                        if 'e'==s1[1] :
                            return False
                        else: pass
                    else:
                        if s1.count(i)==2:
                            if s1.index(i)==0 and 'e' == s1[s1.rindex(i) -1] and '' != s1[s1.rindex(i)+1:]:pass
                            else: return False
                        elif s1.count(i)==1:
                            if 'e' == s1[s1.rindex(i) -1] and '' != s1[s1.rindex(i)+1:]: pass
                            else: return False
                        else:return False
            # 对 “.”处理
            if '.' in s1:
                if s1.count('.') == 1:
                    pass
                else:
                    return False
            # 对 “e”处理
            if 'e' in s1:
                if s1.count('e') == 1:
                    e_l = s1.split('e')
                    if len(e_l[1])>0:
                        if '.' in e_l[1]:
                            return False
                    else:
                        return False

                    if len(e_l[0])>0:
                        if '.'==e_l[0]:
                            return False
                    else:
                        return False
                else:
                    return False
            # 数字判断
            s2 = s1.replace('+','').replace('-','').replace('.','').replace('e','')
            if len(s2)>0:
                for i in s2:
                    if i in ['1','2','3','4','5','6','7','8','9','0']:
                        pass
                    else:return False
                return True
            else:
                return False
```
### 面试题41数据流的中位数
```python
from heapq import *

class MedianFinder:
    def __init__(self):
        self.A = [] # 小顶堆，保存较大的一半
        self.B = [] # 大顶堆，保存较小的一半

    def addNum(self, num: int) -> None:
        if len(self.A) != len(self.B):
            heappush(self.A, num)
            heappush(self.B, -heappop(self.A))
        else:
            heappush(self.B, -num)
            heappush(self.A, -heappop(self.B))

    def findMedian(self) -> float:
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0
```
