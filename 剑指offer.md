# 2.27
###  快速排序
n个元素 平均时间复杂度nlog(n)
基于==分治==的排序
快速排序 以某一元素为基准，划分为两个数组
==划分==算法
 基于左右两个指针同时扫描

### 剑指 Offer 09. 用两个栈实现队列
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

 
```
示例 1：

输入：
["CQueue","appendTail","deleteHead","deleteHead","deleteHead"]
[[],[3],[],[],[]]
输出：[null,null,3,-1,-1]
示例 2：

输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]

```
![avatar](https://pic.leetcode-cn.com/966aebd484002e620d88676847273a061ab9ab6d863ab5079ab347a643461e24-09.gif)
```C++
class CQueue {
public:
     stack<int> stack1;
     stack<int> stack2;

    CQueue() {

    }
    
    void appendTail(int value) {
        stack1.push(value);
    }
    
    int deleteHead() {
        if(stack1.empty()) return -1;
        while(!stack1.empty()){
            int temp = stack1.top();
            stack2.push(temp);
            stack1.pop();
        }
        int res = stack2.top();
        stack2.pop();
        while(!stack2.empty()){
            int temp = stack2.top();
            stack1.push(temp);
            stack2.pop();
        }
        return res;
    }
};
```
双堆栈结构 只使用一个栈 stack1 当作队列，另一个栈 stack2 用来辅助操作。

要想将新加入的元素出现栈底，需要先将 stack1 的元素转移到 stack2，将元素入栈 stack1，最后将 stack2 的元素全部回到 stack1。

stack 堆栈结构，stack.push(value)放入元素 .pop 删除栈顶元素

### 剑指 Offer 30. 包含min函数的栈
定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

 
```
示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```
```c++
class MinStack {
public:
    /** initialize your data structure here. */
        stack<int> stack_x;
        stack<int> min_stack;
    MinStack() {
        min_stack.push(INT_MAX);
    }
    
    void push(int x) {
        stack_x.push(x);
        min_stack.push(::min (min_stack.top(), x));

    }
    
    void pop() {
        stack_x.pop();
        min_stack.pop();
    }
    
    int top() {
        return stack_x.top();
    }
    
    int min() {
        return min_stack.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
 ```
![avatar](https://assets.leetcode-cn.com/solution-static/jianzhi_30/jianzhi_30.gif)
当一个元素要入栈时，我们取当前辅助栈的栈顶存储的最小值，与当前元素比较得出最小值，将这个最小值插入辅助栈中；

当一个元素要出栈时，我们把辅助栈的栈顶元素也一并弹出；

在任意一个时刻，栈内元素的最小值就存储在辅助栈的栈顶元素中。


```  min_stack.push(::min (min_stack.top(), x));```min因为下面重写了，不会调用系统的min函数,所以::min

# 2.28
### 剑指 Offer 06. 从尾到头打印链表
输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
```
输入：head = [1,3,2]
输出：[2,3,1]
```
```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        stack<int> stack;
        while(head){
            stack.push(head->val);
            head = head->next;
        }
        vector<int> res;
        while(!stack.empty()){
            res.push_back(stack.top());
            stack.pop();
        }
        return res;
    }
};
```
简单题 复习链表，数组的用法
链表ListNode

### 剑指 Offer 24. 反转链表
定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```
```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *prev = NULL;
        while(head !=nullptr)
        {
            ListNode *temp = head -> next;
            head -> next = prev;
            prev = head;
            head = temp;
        }
        return prev;
    }
};
```
反转指针
let temp = head.next 用temp存一下当前的head.next
![avatar](https://pic.leetcode.cn/1677407706-mvBIfw-IMG_1164(20230226-182903).PNG)
head.next = prev 把当前的head.next指向prev
![avatar](https://pic.leetcode.cn/1677407770-AcceIJ-IMG_1166(20230226-183036).PNG)
prev = head prev右移到新的头，即现在的head的位置。
![avatar](https://pic.leetcode.cn/1677407926-SDtwyk-IMG_1167(20230226-183227).PNG)

### 剑指 Offer 35. 复杂链表的复制（哈希表 / 拼接与拆分，清晰图解
请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。
```c++
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head == nullptr) return nullptr;
        Node* cur = head;
        unordered_map<Node*, Node*> map;
        // 3. 复制各节点，并建立 “原节点 -> 新节点” 的 Map 映射
        while(cur != nullptr) {
            map[cur] = new Node(cur->val);
            cur = cur->next;
        }
        cur = head;
        // 4. 构建新链表的 next 和 random 指向
        while(cur != nullptr) {
            map[cur]->next = map[cur->next];
            map[cur]->random = map[cur->random];
            cur = cur->next;
        }
        // 5. 返回新链表的头节点
        return map[head];
    }
};

```

# 3.5
### 剑指 Offer 53 - I. 在排序数组中查找数字 I

统计一个数字在排序数组中出现的次数。

 

```示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
示例 2:

输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
```
```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int res = 0;
        for (int i = 0;i<nums.size();i++){
            if(nums[i] == target){
                res++;
            }
        }
        return res;
    }
};
```
太简单 复杂度O(N);
使用二分法 复杂度O(logN)
```c++
class Solution {
public:

    int search(vector<int>& nums, int target) {
        int left =0,right = nums.size()-1;
        int count = 0;
        while(left<right){
            int mid = (left+right)/2;
            if(nums[mid]>=target)
                right=mid;
            if(nums[mid]<target)
                left = mid+1;
        }
        while(left<nums.size()&&nums[left++]==target)
            count++;
        return count;
    }
    
};
```
有序数组的一些操作都应该先考虑二分法O(logn) 而不是遍历O(N)

# 3.6 查找算法 2
### 剑指 Offer 04. 二维数组中的查找
```
在一个 n * m 的二维数组中，每一行都按照从左到右 非递减 的顺序排序，每一列都按照从上到下 非递减 的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

 

示例:

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。

给定 target = 20，返回 false。
```
```c++
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if (matrix.size() == 0 || matrix[0].size() == 0) return false;
        int m = matrix.size(), n = matrix[0].size();
        int x = 0, y = n - 1;
        while (x < m && y >= 0) {
            if (matrix[x][y] == target) {
                return true;
            }
            if (matrix[x][y] > target) {
                --y;
            }
            else {
                ++x;
            }
        }
        return false;
    }
};

```
Z 字形查找

### 剑指 Offer 50. 第一个只出现一次的字符
在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母
```
示例 1:
输入：s = "abaccdeff"
输出：'b'
示例 2:
输入：s = "" 
输出：' '
```
```c++
class Solution {
public:
    int minArray(vector<int>& numbers) {
        int low = 0;
        int high = numbers.size() - 1;
        while (low < high) {
            int pivot = low + (high - low) / 2;
            if (numbers[pivot] < numbers[high]) {
                high = pivot;
            }
            else if (numbers[pivot] > numbers[high]) {
                low = pivot + 1;
            }
            else {
                high -= 1;
            }
        }
        return numbers[low];
    }
};
```
熟悉哈希表操作

# 3.7 搜索与回溯算法
### 剑指 Offer 32 - I. 从上到下打印二叉树
从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
==层序遍历 BFS==
```
例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回：

[3,9,20,15,7]

```
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) {
        vector<int> vec;
        if (root == NULL) { // 初始判空处理
            return vec;
        }

        queue<TreeNode*> que; // 创建队列
        que.push(root); // 加入根节点

        while (!que.empty()) { // 循环队列不为空
            TreeNode* temp = que.front(); // 队首
            que.pop(); // 弹出队首
            if (temp->left) { // 不为空就入队
                que.push(temp->left);
            }
            if (temp->right) {
                que.push(temp->right);
            }
            vec.push_back(temp->val);
        }
        return vec;
    }
};
```
# 3.14 
### 剑指 Offer 26. 树的子结构
```
输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

     3
    / \
   4   5
  / \
 1   2
给定的树 B：

   4 
  /
 1
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

示例 1：

输入：A = [1,2,3], B = [3,1]
输出：false
示例 2：

输入：A = [3,4,5,1,2], B = [4,1]
输出：true
```
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(A == NULL || B == NULL) return false;
        return compare(A,B)||isSubStructure(A->left,B)||isSubStructure(A->right,B);
    }

    bool compare(TreeNode* A , TreeNode* B){
        if(B == NULL) return true;
        if(A == NULL) return false;
        return A->val == B->val && compare(A->left,B->left)&&compare(A->right,B->right);
    }
};
```
先序遍历 + 包含判断

# 3.15 搜索与回溯算法
### 剑指 Offer 27. 二叉树的镜像
请完成一个函数，输入一个二叉树，该函数输出它的镜像。
```
例如输入：

     4
   /   \
  2     7
 / \   / \
1   3 6   9
镜像输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1
```
 

示例 1：

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
if (root == nullptr) return nullptr;
        TreeNode* tmp = root->left;
        root->left = mirrorTree(root->right);
        root->right = mirrorTree(tmp);
        return root;
    }
};
```
这是一道很经典的二叉树问题。显然，我们从根节点开始，递归地对树进行遍历，并从叶子节点先开始翻转得到镜像。如果当前遍历到的节点 root 的左右两棵子树都已经翻转得到镜像，那么我们只需要交换两棵子树的位置，即可得到以 root 为根节点的整棵子树的镜像。

# 3.19
### 剑指 Offer 10- II. 青蛙跳台阶问题
```
一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

示例 1：

输入：n = 2
输出：2
示例 2：

输入：n = 7
输出：21
示例 3：

输入：n = 0
输出：1
```
```c++
class Solution {
public:
    int numWays(int n) {
        int a = 0, b = 1 , sum = 0;
        for (int i = 0; i<=n; i++){
            sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }
};
```
<img src="D:\work\笔记\剑指offer\屏幕截图 2023-03-19 135638.png">

vector for循环可以
```c++
vector<int> vector;
for(int i : vector){
    cout<< i;//i是vector[i]
}

for (declaration : expression){
    //循环体
}
```
其中，两个参数各自的含义如下：
declaration：表示此处要定义一个变量，该变量的类型为要遍历序列中存储元素的类型。需要注意的是，C++ 11 标准中，declaration参数处定义的变量类型可以用 auto 关键字表示，该关键字可以使编译器自行推导该变量的数据类型。
expression：表示要遍历的序列，常见的可以为事先定义好的普通数组或者容器，还可以是用 {} 大括号初始化的序列。

###剑指 Offer 42. 连续子数组的最大和
```
输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

示例1:

输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

```
```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int maxSum = nums[0], tempSum = 0;
        for(int num:nums){
            tempSum += num;
            tempSum = max(tempSum,num);
            if(tempSum>maxSum) maxSum = tempSum;
        } 
        return maxSum;
    }
};
```
==动态规划思想==
f(i) = max( f(i-1)+nums[i] , nums[i])
动态规划就是找此项与上一项的关系

### 剑指 Offer 47. 礼物的最大价值
在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

```
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```
<img src="D:\work\笔记\剑指offer\屏幕截图 2023-03-19 201740.png">

```c++
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> f(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > 0) {
                    f[i][j] = max(f[i][j], f[i - 1][j]);
                }
                if (j > 0) {
                    f[i][j] = max(f[i][j], f[i][j - 1]);
                }
                f[i][j] += grid[i][j];
            }
        }
        return f[m - 1][n - 1];
    }
};
```
注意到状态转移方程中，f(i,j)只会从f(i−1,j)和f(i,j−1)转移而来，而与f(i−2,⋯) 以及更早的状态无关，因此我们同一时刻只需要存储最后两行的状态，即使用两个长度为n的一位数组代替m×n的二维数组f，交替地进行状态转移，减少空间复杂度

```c++
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> f(2, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            int pos = i % 2;
            for (int j = 0; j < n; ++j) {
                f[pos][j] = 0;
                if (i > 0) {
                    f[pos][j] = max(f[pos][j], f[1 - pos][j]);
                }
                if (j > 0) {
                    f[pos][j] = max(f[pos][j], f[pos][j - 1]);
                }
                f[pos][j] += grid[i][j];
            }
        }
        return f[(m - 1) % 2][n - 1];
    }
};
```

### 剑指 Offer 46. 把数字翻译成字符串

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

```
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
```
```c++
class Solution {
public:
    int translateNum(int num) {
        string nums = to_string(num);
        int a = 1, b = 1, sum = 1;
        for(int i=1; i<nums.size(); i++){
            string temp = nums.substr(i-1,2);
            if(temp<="25"&&temp>="10"){
            sum = a+b;
            a = b;
            b = sum;   
            }
            else{
                sum = b;
                a = b;
                b = sum;
            }
        }
        return sum;
    }
};
```
判断字符串的大小可以直接用 > , < 符号，根据字典顺序
状态转移方程 f(i) = f(i-1)+f(i-2) (i-1 与 i-2 组合应该小于25并且==大于10==)
如果不满足上述条件 f(i) = f(i-1)

# 3.20
### 剑指 Offer 48. 最长不含重复字符的子字符串
请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
```
示例 1:

输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

```
```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        // 哈希集合，记录每个字符是否出现过
        unordered_set<char> occ;
        int n = s.size();
        // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        int rk = -1, ans = 0;
        // 枚举左指针的位置，初始值隐性地表示为 -1
        for (int i = 0; i < n; ++i) {
            if (i != 0) {
                // 左指针向右移动一格，移除一个字符
                occ.erase(s[i - 1]);
            }
            while (rk + 1 < n && !occ.count(s[rk + 1])) {
                // 不断地移动右指针
                occ.insert(s[rk + 1]);
                ++rk;
            }
            // 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = max(ans, rk - i + 1);
        }
        return ans;
    }
};
```
判断是否重复首选哈希表，熟悉哈希表操作

### 剑指 Offer 18. 删除链表的节点
给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
返回删除后的链表的头节点。
```
输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
```
```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        ListNode* cur = head->next;
        ListNode* pre = head;
        if(head->val == val) {
            head = head->next;
            return head;
        }
        while(cur->val != val){
            cur = cur->next;
            pre = pre->next;
        }
        pre->next = cur->next;
        return head;
    }
};
```
注意 赋值        
ListNode* cur = head->next;
ListNode* pre = head;
改变cur pre指向，对应head作为头节点也会改
如果想创建一个新list 应该用new list

### 剑指 Offer 22. 链表中倒数第k个节点

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

 

示例：
```
给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.
```
```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode* temp = head;
        for(int i = 0; i<k-1; i++){
            temp = temp->next;
        }
        while(temp->next != NULL){
            temp = temp->next;
            head = head->next;
        }
        return head;
    }
};
```
==巧妙解法==
双指针
快慢指针的思想，第一个指针指向head 第二个指针指向第k-1个，两个指针同时向后遍历，到第二个指针指向最后一个，第一个指针也就指向倒数第k个了

### 剑指 Offer 25. 合并两个排序的链表
输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```
```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* prehead = new ListNode(-1);
        ListNode* prev = prehead;
        while(l1 != NULL && l2 != NULL){
            if(l1->val <= l2->val){
                prev->next = l1;
                l1 = l1->next;
            }
            else{
                prev->next = l2;
                l2 = l2->next;
            }
            prev = prev->next;
        }
        prev->next = l1 == NULL? l2:l1;
        return prehead->next;
    }
};
```
![avatar](https://assets.leetcode-cn.com/solution-static/jianzhi_25/19.PNG)
我们设定一个哨兵节点prehead
注意建立一个新的LIST时应该ListNode* prehead = new ListNode(-1);

### 剑指 Offer 52. 两个链表的第一个公共节点
输入两个链表，找出它们的第一个公共节点
![avatar](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }
        ListNode* pA = headA;
        ListNode* pB = headB;
        while(pA != pB){
            if(pA != NULL) pA = pA->next;
            else pA = headB;
            if(pB != NULL) pB = pB->next;
            else pB = headA;
        }
        return pA;
    }
};
```
<img src="D:\work\笔记\剑指offer\屏幕截图 2023-03-20 215512.png">

# 3.21
### 剑指 Offer 57. 和为s的两个数字
输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
```
输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
```
双指针，因为数组中元素是增序的，则可采用两端逼近原则
i与j分别为左右两个指针
1.下标为i、j的两元素和等于target，则下表为i、j的元素就是所找数
2.下标为i、j的两元素和小于target，则i++
3.下标为i、j的两元素和大于target，则j--
```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int left = 0, right = nums.size()-1;
        vector<int> res;
        while(left < right){
            if(nums[left] + nums[right] > target) right--;
            else if(nums[left] + nums[right] < target) left++;
            else {
                res = {nums[left],nums[right]};
                return res;
            }
        }
        return res;
    }
};
```
### 剑指 Offer 58 - I. 翻转单词顺序
输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。
```
示例 1：

输入: "the sky is blue"
输出: "blue is sky the"
示例 2：

输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
示例 3：

输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

```c++
class Solution {
public:
    string reverseWords(string s) {
        reverse(s.begin(),s.end());
        string res,temp;
        int i = 0;
        while(i < s.size()){
            if(s[i] == ' ') {
                i++;
                continue;
            }
            while(s[i] != ' '&& i < s.size()){
                temp += s[i];
                i++;
            }
            reverse(temp.begin(),temp.end());
            if(!res.empty())res+=' ';
            res+=temp;
            temp.clear();

            i++;
        }
        return res;
    }
};
```

![avatar](https://assets.leetcode-cn.com/solution-static/jianzhi_58_I/reverse_whole2.png)
先直接倒转之后再每个单词再反一次

### 剑指 Offer 12. 矩阵中的路径
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

例如，在下面的 3×4 的矩阵中包含单词 "ABCCED"（单词中的字母已标出）。
![avatar](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

```c++
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        rows = board.size();
        cols = board[0].size();
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }
private:
    int rows, cols;
    bool dfs(vector<vector<char>>& board, string word, int i, int j, int k) {
        if(i >= rows || i < 0 || j >= cols || j < 0 || board[i][j] != word[k]) return false;
        if(k == word.size() - 1) return true;
        board[i][j] = '\0';
        bool res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) || 
                      dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i , j - 1, k + 1);
        board[i][j] = word[k];
        return res;
    }
};

```
dfs深度优先遍历

### 面试题13. 机器人的运动范围
广度优先算法
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

示例 1：
输入：m = 2, n = 3, k = 1
输出：3
![avatar](https://pic.leetcode-cn.com/1603024999-XMpudY-Picture9.png)

数位和增量公式：设x的数位和为sx
当(x+1)%10 = 0时，sx+1 = sx - 8; 19,20的数位和为10和2；
当(x+1)%10 ！= 时，sx+1 = sx +1; 18,19的数位和为9和10。

#### 方法一：深度优先遍历 DFS
深度优先搜索： 可以理解为暴力法模拟机器人在矩阵中的所有路径。DFS 通过递归，先朝一个方向搜到底，再回溯至上个节点，沿另一个方向搜索，以此类推。
剪枝： 在搜索中，遇到数位和超出目标值、此元素已访问，则应立即返回，称之为 可行性剪枝 。
算法解析：
递归参数： 当前元素在矩阵中的行列索引 i 和 j ，两者的数位和 si, sj 。
终止条件： 当 ① 行列索引越界 或 ② 数位和超出目标值 k 或 ③ 当前元素已访问过 时，返回 
0 ，代表不计入可达解。
递推工作：
1. 标记当前单元格 ：将索引 (i, j) 存入 Set visited 中，代表此单元格已被访问过。
2. 搜索下一单元格： 计算当前元素的 下、右 两个方向元素的数位和，并开启下层递归 。
回溯返回值： 返回 1 + 右方搜索的可达解总数 + 下方搜索的可达解总数，代表从本单元格递归搜索的可达解总数。

```c++
class Solution {
public:
    int movingCount(int m, int n, int k) {
        vector<vector<bool>> visited(m, vector<bool>(n, 0));
        return dfs(0, 0, 0, 0, visited, m, n, k);
    }
private:
    int dfs(int i, int j, int si, int sj, vector<vector<bool>> &visited, int m, int n, int k) {
        if(i >= m || j >= n || k < si + sj || visited[i][j]) return 0;
        visited[i][j] = true;
        return 1 + dfs(i + 1, j, (i + 1) % 10 != 0 ? si + 1 : si - 8, sj, visited, m, n, k) +
                   dfs(i, j + 1, si, (j + 1) % 10 != 0 ? sj + 1 : sj - 8, visited, m, n, k);
    }
};
```
#### 方法二：广度优先遍历 BFS
BFS/DFS ： 两者目标都是遍历整个矩阵，不同点在于搜索顺序不同。DFS 是朝一个方向走到底，再回退，以此类推；BFS 则是按照“平推”的方式向前搜索。
BFS 实现： 通常利用队列实现广度优先遍历。
算法解析：
初始化： 将机器人初始点(0,0) 加入队列 queue ；
迭代终止条件： queue 为空。代表已遍历完所有可达解。
迭代工作：
1. 单元格出队： 将队首单元格的 索引、数位和 弹出，作为当前搜索单元格。
2. 判断是否跳过： 若 ① 行列索引越界 或 ② 数位和超出目标值 k 或 ③ 当前元素已访问过 时，执行 continue 。
3. 标记当前单元格 ：将单元格索引 (i, j) 存入 Set visited 中，代表此单元格 已被访问过 。
4. 单元格入队： 将当前元素的 下方、右方 单元格的 索引、数位和 加入 queue 。
返回值： Set visited 的长度 len(visited) ，即可达解的数量。
```c++
class Solution {
public:
    int movingCount(int m, int n, int k) {
        vector<vector<bool>> visited(m, vector<bool>(n, 0));
        int res = 0;
        queue<vector<int>> que;
        que.push({ 0, 0, 0, 0 });
        while(que.size() > 0) {
            vector<int> x = que.front();
            que.pop();
            int i = x[0], j = x[1], si = x[2], sj = x[3];
            if(i >= m || j >= n || k < si + sj || visited[i][j]) continue;
            visited[i][j] = true;
            res++;
            que.push({ i + 1, j, (i + 1) % 10 != 0 ? si + 1 : si - 8, sj });
            que.push({ i, j + 1, si, (j + 1) % 10 != 0 ? sj + 1 : sj - 8 });
        }
        return res;
    }
};
```

# 3.23 
### 剑指 Offer 34. 二叉树中和为某一值的路径
给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
叶子节点 是指没有子节点的节点。
![avatar](https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg)
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]

深度优先遍历
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        dfs(root, target);
        return res;
    }
    void dfs(TreeNode* root, int target){
        if(root == nullptr) return;
        target -= root->val;
        path.emplace_back(root->val);
        if(root->left == nullptr && root->right == nullptr && target == 0) {
            res.emplace_back(path);
        }
            dfs(root->left,target);
            dfs(root->right,target); 
            path.pop_back();
    }
};
```
算法流程：
pathSum(root, sum) 函数：

初始化： 结果列表 res ，路径列表 path 。
返回值： 返回 res 即可。
recur(root, tar) 函数：

递推参数： 当前节点 root ，当前目标值 tar 。
终止条件： 若节点 root 为空，则直接返回。
递推工作：
1. 路径更新： 将当前节点值 root.val 加入路径 path ；
2. 目标值更新： tar = tar - root.val（即目标值 tar 从 sum 减至0）；
3. 路径记录： 当 ① root 为叶节点 且 ② 路径和等于目标值 ，则将此路径 path 加入 res 。
4. 先序遍历： 递归左 / 右子节点。
5. ==路径恢复==： 向上回溯前，需要将当前节点从路径 path 中删除，即执行 path.pop()

### 剑指 Offer 36. 二叉搜索树与双向链表
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。
==二叉搜索树==： 左节点<根节点<右节点
为了让您更好地理解问题，以下面的二叉搜索树为例：
![avatar](https://assets.leetcode.com/uploads/2018/10/12/bstdlloriginalbst.png)
我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。
下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。
![avator](https://assets.leetcode.com/uploads/2018/10/12/bstdllreturndll.png)
特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。

==中序遍历==
![avatar](https://pic.leetcode-cn.com/1599401091-PKIjds-Picture1.png)
中序遍历 为对二叉树作 “左、根、右” 顺序遍历，递归实现如下：
```c++
// 打印中序遍历
void dfs(Node* root) {
    if(root == nullptr) return;
    dfs(root->left); // 左
    cout << root->val << endl; // 根
    dfs(root->right); // 右
}
```
算法流程：
dfs(cur): 递归法中序遍历；
1. 终止条件： 当节点 cur 为空，代表越过叶节点，直接返回；
2. 递归左子树，即 dfs(cur.left) ；
3. 构建链表：
 3.1 当 pre 为空时： 代表正在访问链表头节点，记为 head ；
 3.2 当 pre 不为空时： 修改双向节点引用，即 pre.right = cur ， cur.left = pre ；
 3.3 保存 cur ： 更新 pre = cur ，即节点 cur 是后继节点的 pre ；
4. 递归右子树，即 dfs(cur.right) ；
treeToDoublyList(root)：
1. 特例处理： 若节点 root 为空，则直接返回；
2. 初始化： 空节点 pre ；
3. 转化为双向链表： 调用 dfs(root) ；
4. 构建循环链表： 中序遍历完成后，head 指向头节点， pre 指向尾节点，因此修改 head 和 pre 的双向节点引用即可；
5. 返回值： 返回链表的头节点 head 即可；
![avatar](https://pic.leetcode-cn.com/1599402776-WMHCrE-Picture7.png)

```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
public:
    Node* treeToDoublyList(Node* root) {
    if (root == nullptr) return nullptr;
    dfs(root);
    head->left = pre;
    pre->right = head;
    return head;    
    }
    Node *pre,*head;
    void dfs(Node* cur){
        if(cur == nullptr) return;
        dfs(cur->left);
        if(pre != nullptr) pre->right = cur;
        else head = cur;
        cur->left = pre;
        //cout<<cur-val<<endl;
        pre = cur;
        dfs(cur->right);
    }
};
```

# ==快速排序==（熟记全文并背诵）
```c++
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        int left = 0, right = nums.size()-1;
        quicksort(nums,left,right);
        return nums;
    }
    void quicksort(vector<int>& nums, int begin, int end){
        if(begin>=end) return; 
        int left = begin, right = end;
        int temp = nums[left];
        while(left<right){
            while(left<right && temp <= nums[right]) right--;
            nums[left] = nums[right];
            while(left<right && temp >= nums[left]) left++;
            nums[right] = nums[left];

        }
        nums[left] = temp;
        quicksort(nums,begin,left-1);
        quicksort(nums,left+1,end);
    }
};
```

## 随机快速排序
```c++
    int partition(vector<int>& nums, int l, int r) {
        int pivot = nums[r];
        int i = l - 1;
        for (int j = l; j <= r - 1; ++j) {
            if (nums[j] <= pivot) {
                i = i + 1;
                swap(nums[i], nums[j]);
            }
        }
        swap(nums[i + 1], nums[r]);
        return i + 1;
    }
    int randomized_partition(vector<int>& nums, int l, int r) {
        int i = rand() % (r - l + 1) + l; // 随机选一个作为我们的主元
        swap(nums[r], nums[i]);
        return partition(nums, l, r);
    }
    void randomized_quicksort(vector<int>& nums, int l, int r) {
        if (l < r) {
            int pos = randomized_partition(nums, l, r);
            randomized_quicksort(nums, l, pos - 1);
            randomized_quicksort(nums, pos + 1, r);
        }
    }
public:
    vector<int> sortArray(vector<int>& nums) {
        srand((unsigned)time(NULL));
        randomized_quicksort(nums, 0, (int)nums.size() - 1);
        return nums;
    }
};

```

