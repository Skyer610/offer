##vector常用的成员函数
vector是最常用的容器之一，功能十分强大，可以储存、管理各种类型的数据。vector也可以称为动态数组，因为其大小是根据实时更新而变化的，正因为如此vector显得更加灵活易用。

储存int型的值 vector<int> v;

储存double型的值 vector<double> v;

储存string型的值 vector<string> v;

储存结构体或者类的值的值 vector<结构体名> v;

当然也可以定义vector数组，n为数组的大小;

储存int型的值 vector<int> v[n];

储存double型的值 vector<double> v[n];

```c++
vector.size(); //返回返回容器中元素个数

vector.begin(); //返回头部迭代器

vector.end(); //返回尾部+1迭代器

vector.rbegin(); //返回逆首部迭代器

vector.rend(); //返回逆尾部-1迭代器

vector.front(); //返回首个元素

vector.back(); //返回尾部元素

vector.push_back(); //在末尾添加一个元素

vector.emplace_back(); //和push_back()是一样的作用

vector.pop_back(); //弹出最后一个元素

vector.empty(); //判断是否为空

vector.insert(); //在指定位置插入元素

vector.erase(); //在指定位置删除元素

vector.clear(); //清空容器

swap(a[1],a[2]);//交换两个数组
```

vector.begin()返回的的是向量的头指针，指向第一个元素
```c++
vector<int>a={1,0};
vector<int>::iterator iter=a.begin();
cout<<*iter;  //输出1
```


## queue常用函数
queue是一种容器转换器模板，调用#include<queue>即可使用队列类。

queue<Type, Container> (<数据类型，容器类型>）。初始化时必须要有数据类型，容器可省略，省略时则默认为 deque 类型。

不能用vector容器初始化queue

因为queue转换器要求容器支持front（）、back（）、push_back（）及 pop_front（），说明queue的数据从容器后端入栈而从前端出栈。所以可以使用deque和list对queue初始化，而vector因其缺少pop_front（），不能用于queue。

```c++
queue.push(); // 在队尾插入一个元素

queue.pop(); // 删除队列第一个元素

queue.size(); // 返回队列中元素个数

queue.empty(); // 如果队列空则返回true

queue.front(); // 返回队列中的第一个元素

queue.back(); // 返回队列中最后一个元素

swap(a,b);      //交换两个队列

```
## 字符串（String）
String是STL中的一个重要的部分，主要用于字符串处理。可以使用输入输出流方式直接进行String读入输出，类似于C语言中的字符数组，由C++的算法库对String类也有着很好的支持，大多时候字符串处理的问题使用String要比字符数组更加方便

创建String类型变量

String s;直接创建一个空的（大小为0）的String类型变量s
String s = *char;创建String时直接用字符串内容对其赋值，注意字符串要用双引号“”
String s(int n,char c);创建一个String，由n个c组成，注意c是字符型要用单括号‘ ’
读入String

cin>>s;读入s，遇到空格或回车停止，无论原先s是什么内容都会被新读入的数据替代
getline(cin,s)；读入s,空格也同样会读入，直到回车才会停止
输出String

cout<<s；将s全部输出到一行（不带回车）
赋值、比较、连接运算符：
赋值运算符：=将后面的字符串赋值给前面的字符串O(n)

比较运算符：== != < <= > >=比较的是两个字符串的字典序大小O(n)

连接运算符：+ +=将一个运算符加到另一个运算符后面O(n)

s[index]返回字符串s中下标为index的字符，String中下标也是从0开始O(1)

s.substr(p,n)返回从s的下标p开始的n个字符组成的字符串，如果n省略就取到底O(n)

s.length()返回字符串的长度O(1)

s.empty()判断s是否为空，空返回1,不空返回0,O(1)

s.erase(p0,len)删除s中从p0开始的len个字符，如果len省略就删到底O(n)

s.erase(s.begin()+i)删除下标为i个字符O(n)

s1.insert(p0,s2,pos,len)后两个参数截取s2,可以省略O(n)

s.insert(p0,n,c)在p0处插入n个字符c O(n)

s1.replace(p0,len0,s2,pos,len)删除p0开始的len0个字符，然后在p0处插入串s2中从pos开始的len个字符，后两个参数可以省略O(n)

s1.find（s2,pos）从前往后，查找成功时返回第一次出现的下标，失败返回string::npos的值（-1）O(n*m)

s1.rfind(s2，pos)从pos开始从后向前查找字符串s2中字符串在当前串后边第一次出现的下标O(n*m)

## 集合（set）
集合（set）是一种包含对象的容器，可以快速地（logn）查询元素是否在已知几集合中。
set中所有元素是有序地，且只能出现一次，因为set中元素是有序的，所以存储的元素必须已经定义过[<]运算符（因此如果想在set中存放struct的话必须手动重载[<]运算符，和优先队列一样）
与set类似的还有：

multiset元素有序可以出现多次
unordered_set元素无序只能出现一次
unordered_multiset元素无序可以出现多次
博客推荐：https://www.cnblogs.com/zyxStar/p/4542835.html

集合（set）
建立方法：
*set<Type> s;
multiset<Type> s;
unorded_set<Type> s;
unorded_multiset<Type> s;
如果Type无法进行比较，还需要和优先队列一样定义<运算符

遍历方法：
for (auto i:s)cout<< i <<" ";
//和vector的类似

使用方法：

s.insert(item)：在s中插入一个元素 O(logn)

s.size()：获取s中元素个数 O(1)

s.empty()：判断s是否为空 O(1)

s.clear()：清空s O(n)

s.find(item)：在s中查找一个元素并返回其迭代器，找不到的话返回s.end() O(logn)

s.begin()：返回s中最小元素的迭代器，注意set中的迭代器和vector中的迭代器不同，无法直接加上某个数，因此要经常用到–和++ O(1)

s.end()：返回s中最大元素的迭代器的后一个迭代器 O(1)

s.count(item)：返回s中item的数量。在set中因为所有元素只能在s中出现一次，所以返回值只能是0 或1，在multiset中会返回存的个数 O(logn)

s.erase(position)：删除s中迭代器position对应位置的元素 O(logn)

s.erase(item)：删除s中对应元素 O(logn)

s.erase(pos1，pos2)：删除[pos1,pos2]这个区间的位置的元素 O(logn+pos2-pos1)

s.lower_bound(item)：返回s中第一个大于等于item的元素的迭代器，找不到就返回s.end() O(logn)

s.upper_bound(item)：返回 s中第一个大于item的元素的迭代器，找不到就返回s.end() O(logn)

## 映射（map）
map是照特定顺序存储由key和value的组合形成的元素的容器，map中元素按照key进行排序，每个key都是唯一的，并对应着一个value,value可以重复。
map的底层实现原理与set一样都是红黑树。
与map类似的还有unordered_map,区别在于key不是按照顺序排序


建立方法：
```
map<key,value> mp;
unordered_map<key,value> mp;
```

遍历方法：
```
for(auto i:mp);
cout<<i.first<<’ '<<i.second<<endl;
```

map的常用函数：

mp.size()：获取mp中元素个数 O(1)

mp.empty()：判断mp是否为空 O(1)

mp.clear()：清空mp O(1)

mp.begin()：返回mp中最小key的迭代器，和set一样，只可以用到–和++操作 O(1)

mp.end()：返回mp中最大key 的迭代器的后一个迭代器 O(1)

mp.find(key)：在mp中查找一个key 并返回其iterator,找不到的话返回s.end() O(logn)

mp.count(key)：在mp中找key的数量，因为map中key唯一，所以只会返回0或1 O(logn)

mp.erase(key)：在 mp 中删除key 所在的项，key和value都会被删除 O(logn)

mp.lower_bound(item)：返回mp中第一个key大于等于item的迭代器，找不到就返回mp.end()0(logn)

mp.upper_bound(item)：返回 mp中第一个key大于item的迭代器，找不到就返回mp.end() O(logn)

mp[key]：返回 mp中 key 对应的 value。如果key 不存在，则返回 value 类型的默认构造器(defaultconstructor)所构造的值，并该键值对插入到mp中 O(logn)

mp[key] = xxx：如果mp中找不到对应的key则将键值对(key：xxx)插入到mp中，如果存在key则将这个key对应的值改变为xxx O(logn)

unordered的作用：
set和map都可以在前面加上unorder_使得内部的元素改成不按顺序存储的，其余的功能都不改变，虽然无法顺序存储，但底层原理是hash，可以使得所有的查询、修改、删除操作都变成O(1)复杂度

