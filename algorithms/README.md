# 数据结构和算法

## 数据结构和算法的关系
* 要解决的问题是什么？例如，排序问题
* 用什么样的算法?  例如，10+种的排序算法
    1. 时间复杂度：平均，最好，最坏
    1. 空间复杂度
    1. 内存使用方式？
    1. 稳定性
* 算法需要的数据结构支撑？ 例如，每种排序算法依赖的数据结构？
    1. 应用场景
    1. 增删改查


## 数据结构
1. LinkList 链表
    * 固定数组的问题：最大值确定，浪费空间，插入和删除耗时
1. 栈
1. 队列
1. 集合
1. [hash 哈希表](https://segmentfault.com/a/1190000022679511)
1. [树](https://zhuanlan.zhihu.com/p/90255760)
    1. 无序树
    1. 有序树
    1. 二叉树
        1. 满二叉树
        1. 完全二叉树
        1. 完满二叉树
        1. 二叉查找树(二叉搜索树，二叉排序树，BST)
        1. 平衡二叉树
    1. 霍夫曼树
    1. AVL树
    1. 红黑树
    1. 伸展树
    1. 替罪羊树 ？
    1. B树
    1. B+树
    1. B*树
    1. 字典树
    1. 线索二叉树
1. 堆
1. 优先队列
1. 图
1. [SkipList 跳表](https://github.com/HiWong/SkipListPro)
1. [bitmap 位图](https://www.cnblogs.com/dragonsuc/p/10993938.html)
1. [GEOHash](https://zhuanlan.zhihu.com/p/35940647)
2. bloomfilter, 布隆过滤器

## 算法
1. [排序算法](https://www.runoob.com/w3cnote/ten-sorting-algorithm.html)
    1. 冒泡 o(n*n), 两两比较
    2. 选择 O(n*n)，最大最小值放置在首尾
    3. 插入 O(n*n)，
    4. 希尔 O(n*logn)
    5. 归并merge sort O(n*logn)
    6. 快排 
    7. 堆排序
    8. 计数排序
    9. 桶排序
    10. 基数排序 
2. 查找
    1. 顺序查找
    2. 二分查找
    3. 插值查找
    4. 斐波那契查找
    5. 树表查找
    6. 分块查找
    7. 哈希查找
3. [集合求交](https://blog.csdn.net/csdn_zaw/article/details/106926041)
    1. 2层for循环，O(n*n)
    2. 拉链法 O(n)
    3. 分桶并行
    4. bitmap求交,『与』操作, 连续内存空间占用
    5. skiplist (与分桶的区别)
4. 去重问题
    1. hashtable,hash,simhash
    2. bloomfilter
5. 最短路径


## 开源参考
1. c实现
    * https://github.com/fragglet/c-algorithms
2. c++ 实现 
    * https://github.com/mandliya/algorithms_and_data_structures
    * https://github.com/xtaci/algorithms
