一、基础语法
变量的声明和定义有什么区别
写出 bool 、int、 float、指针变量与“零值”比较的 if 语句
short i = 0; i = i + 1L；这两句有错吗
&&和&、||和|有什么区别
sizeof 和 strlen 的区别
C 语言的关键字 static 和 C++ 的关键字 static 有什么区别
写一个“标准”宏 MIN
简述 strcpy、sprintf 与 memcpy 的区别
typedef 和 define 有什么区别
关键字 const 是什么，const 作用及用法
extern 有什么作用
编码实现某一变量某位清 0 或置 1
流操作符重载为什么返回引用
编码实现字符串转化为数字
用 C 编写一个死循环程序
谈谈你对编程规范的理解或认识
中断函数
7、四种cast，智能指针
1. sizeof 和 strlen 的区别
2. lambda 表达式（匿名函数）的具体应用和使用场景
3. explicit 的作用（如何避免编译器进行隐式类型转换）
5. static 的作用， 在类中使用的注意事项（定义、初始化和使用），全局变量和普通全局变量的异同
9. define 和 const 的区别
10. define 和 typedef 的区别
11. 用宏实现比较大小，以及两个数中的最小值
12. inline 作用及使用方法，函数工作原理
14. 宏定义（define）和内联函数（inline）的区别
4. C 和 C++ static 的区别,struct 的区别？
21. 为什么有了 class 还保留 struct, class 和 struct 的异同
22. struct 和 union 的区别
24. volatile 的作用？是否具有原子性，对编译器有什么影响？
25. 什么情况下一定要用 volatile， 能否和 const 一起使用？
26. 返回函数中静态变量的地址会发生什么？
27. extern C 的作用？
28. sizeof(1==1) 在 C 和 C++ 中分别是什么结果？
29. memcpy 函数的底层原理？
30. strcpy 函数有什么缺陷？
31. auto 类型推导的原理
1、const、static作用。
程序编译的过程
计算机内部如何存储负数和浮点数？
函数调用的过程？
左值和右值
17、动态库和静态库的区别。
2、编译原理，尝试自己写过语言或语言编译器。


二、指针和引用
a 和&a 有什么区别
一个指针可以是 volatile 吗
C++的引用和 C 语言的指针有什么区别
设置地址为 0x67a9 的整型变量的值为 0xaa66
简述指针常量与常量指针区别
数组名和指针的区别
如何避免“野指针”
常引用有什么作用
指针参数传递和引用参数传递
形参与实参的区别
从汇编层去解释一下引用
10、指针和引用作用以及区别。
从汇编层去解释一下引用
11、c++11用过哪些特性，auto作为返回值和模板一起怎么用，函数指针能和auto混用吗。
内存有哪几种类型
简述 C、C++程序编译的内存分配情况
Ｃ中的 malloc 和Ｃ＋＋中的 new 有什么区别
内存泄漏？面对内存泄漏和指针越界，你有哪些方法？你通常采用哪些方法来避免和减少这类错误？
4、malloc、free和new、delete区别，引出malloc申请大内存、malloc申请空间失败怎么办。
15. new 的作用？
16. new 和 malloc 如何判断是否申请到内存？
17. delete 实现原理？delete 和 delete[] 的区别？
18. new 和 malloc 的区别，delete 和 free 的区别 ，malloc 的原理？malloc 的底层实现？


三、面向对象
面向对象的三大特征
C++的空类有哪些成员函数
谈谈你对拷贝构造函数和赋值运算符的认识
构造函数能否为虚函数
谈谈你对面向对象的认识
用 C++设计一个不能被继承的类
访问基类的私有虚函数
简述类成员函数的重写、重载和隐藏的区别
简述多态实现的原理
2、c++面向对象三大特征及对他们的理解，引出多态实现原理、动态绑定、菱形继承。
3、虚析构的必要性，引出内存泄漏，虚函数和普通成员函数的储存位置，虚函数表、虚函数表指针。

设计模式: 单例、工厂模式、代理、适配器、模板，使用场景。

四、数据结构和算法
链表和数组有什么区别
怎样把一个单链表反序
简述队列和栈的异同
堆和自由存储区的区别
能否用两个栈实现一个队列的功能
计算一颗二叉树的深度
排序：
直接插入排序
冒泡排序
直接选择排序
堆排序
基数排序
在二元树中找出和为某一值的所有路径
5、stl熟悉吗，vector、map、list、hashMap，vector底层，map引出红黑树。优先队列用过吗，使用的场景。无锁队列听说过吗，原理是什么（比较并交换）
6、实现擅长的排序，说出原理（快排、堆排）

STL标准库
3、泛型模板实用度高。

五、多线程
15、进程间通信。会选一个详细问。
16、多线程，锁和信号量，互斥和同步。
9、进程和线程区别。


六、其他
1、tcp和udp区别
1、boost用过哪些类，thread、asio、signal、bind、function
1、QT信号槽实现机制，QT内存管理，MFC消息机制。
1、针对网络框架（DPDK）、逆向工程（汇编）、分布式集群（docker、k8s、redis等）、CPU计算（nvidia cuda)、图像识别（opencv、opengl、tensorflow等)、AI等有研究。
1、提高c++性能，你用过哪些方式去提升（构造、析构、返回值优化、临时对象（使用operator=()消除临时对象）、
内联（内联技巧、条件内联、递归内联、静态局部变量内联）、内存池、使用函数对象不使用函数指针、编码（编译器优化、预先计算）、
设计（延迟计算、高效数据结构）、系统体系结构（寄存器、缓存、上下文切换））。


