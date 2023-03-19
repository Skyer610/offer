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
  