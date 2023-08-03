# 一些奇奇怪怪的想法
enough talk,show me the code !

## 一些想法
1. 网页提取
传统采用的xpath，csspath的方式从网页中提取结构化文本，需要有一定html知识，如果像gpt那样，给定一个网页的html源码，再给出想要提取哪些内容，按照什么格式输出，就能完成任务，效率可提高不少。
    * 广泛抓取web页的html源码，用gpt2的方式准备预训练模型；
    * 准备一些监督数据，模型微调， 输入一个网页内容，输出json格式的文本。
    * hfrl？
用途：
    1. 结构化数据提取
    2. 正文提取
2. OCR
视觉方式读取图像，区分识别文本、图片等， 转为html等格式，图片去水印？

3. 两个相邻的段落，是不是应该被合成为一个段落？
从pdf中复制文本，粘贴成纯文本时，经常会把段落结构打乱。需要有个模型，讲他们合并为正确的段落结构。

4. 可视化的时间线
在读历史时候，同一时期各条线上的人物经历脉络，重要时间节点上，中外历史对比等，有个直观的时间线可视化的对比，会很有帮助。

5. 结合chatGPT, 《易经》占卜

## Prompts
1. 图片和周围的文本，图片-文字能建立映射关系的实体？ 例如：图片中有人物，将文字中的人物名称提取出来，和图像中的人脸建立映射关系。

https://www.ruanyifeng.com/blog/2020/08/rsync.html

## 索引
* [数据结构和算法](./algorithms/README.md)
* [深度学习](./deep_learning/README.md)
* [数字孪生](./digital_twins/README.md)
* [Arduino](./robotech/ardunio.md)
* [架构](./architecture/README.md)
* [swift_playground](./swift_playground/)
* [stocks](./stocks/README.md)

## 两种方式
1. 先定义问题，再寻找工具
    1. 问题是什么？例如排序问题、查找问题、集合求交问题等
    1. 用什么样的算法? 例如，10+种的排序算法
    1. 算法需要的数据结构支撑？ 例如，每种排序算法依赖的数据结构不一样
2. 先介绍工具，再说该工具能干啥，举一反三
* 有点像物理学和数学的关系，物理学提出问题，数学提供解决问题的工具；
* 教授或学习时，问题和工具应经常建立连接关系；
* 使用合适的工具解决特定的问题，思维保持开放，避免出现拿着锤子看什么都是钉子或者什么问题都能一锤子解决的心态。

## 服务器端编程
1. 互联网，游戏都要用到server端编程
1. 通信协议：http,rpc等
1. 关系型数据库:sql, 事务型:MySQL,PostgreSQL; 分析型：Clickhouse
1. kv数据库:redis,memcached,monodb等
1. 倒排索引库
1. 语义索引库

## 大数据
1. hadoop/spark
1. clickhouse


## 编程语言
* [python]()
* [c/c++](./c_cpp/README.md)
* [c#] Windows客户端，Unity3D
* [javascript] 界面前端

## 输出
* [Markdown数学公式](https://blog.csdn.net/weixin_42782150/article/details/104878759)

