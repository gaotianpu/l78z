# 数字孪生
做一款家居装修软件，普通人可以轻松设计自己的装修方案，角色可以在虚拟世界体验装修效果，修正一些不合理的方案，像玩3D游戏一样简单。房间每个物品都和实际电商报价做真实对应，可以知道真实的成本有多少。

## 一、生态：
1. 房地产开发商、中介房子的3D建模信息
1. 建材、家具、家电等提供摆放物件的3D建模
1. 专业的装修设计者提供样板间
1. 普通用户在样板间的基础上改动房间布局
1. 数字孪生概念：门铃监控视频等，房间灯是否亮着，空调是否开着，煤气灶是否燃烧着，房间温度、湿度、光线，空气清洁程度等
1. 普通人可以在这个虚拟场景里做什么？如何增加趣味性和普适性？
    1. 私有房间？
    1. 公共服务器？

## 二、技术栈：
1. 3D建模: 如何做到高效低成本的渲染能力
    1. 根据贝壳网提供的户型图、房间照片、vr等信息，自动为房间建模
    1. 针对房间环境建模，有一台多线激光雷达+照相机，可以自助移动，自动扫描房间的3D点云信息，能够区分强体和房间内的物品，简单修改后可生成3D场景信息，并能一键清楚房间内已有物品。
    1. 家具物品的3D扫描，手持的、固定的？3D几何结构+材质渲染信息
    1. 家庭成员的3D建模
1. 用户交互
    1. 为专业用户提供更多的设计工具？
    1. 为普通用户提供类似游戏场景，可简单执行替换、摆放等操作
1. 物联网接入
    1. 水电气器材的状态
    1. 房间的人类宠物感知、光线强度、温度、湿度、空气质量，瓦斯监控等
1. 智能化
    1. 装修风格，家具自动摆放？

## 三、需要哪些技术：
    1. 手工建模
        1. 游戏影视动画,blender
        1. 建筑园林,sketchup
        1. 工业设计,freecad
    2. 自动化建模
        1. 激光雷达
        2. 景深相机
        3. 3D扫描仪
        4. 传感器
        4. 高精地图
        5. 模拟仿真
    2. 交互
        1. unity3D，unreal, roblox studio?
        2. 物理引擎，碰撞检测
    3. 物联网
        1. 各类传感器数据接入
    4. 思考:深度学习能在3D世界中发挥什么样的作用？
        1. 利用GAN生成，各类艺术风格人物、物品？

## 四、从哪里开始？
1. 学习blender/sketchup，视频教程1.5倍速观看，了解3D建模的整体流程
2. 学习unity，视频教程1.5倍速观看，了解制作一款3D游戏的整体流程
2. roblox studio的一些启发(国内对应的《重启世界》)
    1. 简化版的 unity
    1. 封装了很多常用的物理引擎组件
    1. 玩家系统：玩家设置预置几种类型，简单修改参数就能用,相机和玩家自动绑定，前后左右移动，跳跃，下蹲，旋转，击发等可以抽象出来。
    1. 游戏效果预览机制很直观
    1. 基于roblox的一些游戏玩法很有借鉴意义
3. 目前市面上的装修软件调研
4. 制作一款demo，给定的空间，少量的家具物品，用户可以搜索添置摆放家具物品
5. 市面上的室内场景构建、3D扫描建模方案调研
6. 根据户型图生成一个简单的房屋3D模型？
    1. blender支持哪些3D格式导入？
    1. 学习obj格式：
        1. 使用blender制作简单模型，导出obj格式，学习obj格式语法；
            1. [OBJ格式模型详细介绍](https://www.cnblogs.com/linuxAndMcu/p/14483146.html)
        2. 使用python脚本生成obj文件，导入至blender测试效果；
        3. 其他obj加载展示工具：js,python,c++,c#实现版本？
            1. [js](https://github.com/mrdoob/three.js)
            1. [PyWavefront](https://github.com/pywavefront/PyWavefront)
            1. [c++ opengl](http://www.opengl-tutorial.org/cn/beginners-tutorials/tutorial-7-model-loading/)
        3. 使用blender制作稍微复杂点的房屋3d模型，查看obj格式；使用python生成并导入测试效果 
        4. 加入材质渲染效果的模型文件格式？
    1. 贝壳网等户型图信息提取
    1. 根据户型图信息生成房屋3D模型
    1. 根据房间的照片信息，自动渲染？有点难度

Web 3D 
1. [tree.js](https://threejs.org/)
1. [WebGL Load Obj](https://webglfundamentals.org/webgl/lessons/webgl-load-obj.html)
1. [3D games on the Web](https://developer.mozilla.org/zh-CN/docs/Games/Techniques/3D_on_the_web)

问题：
1. 模型设计时，用4边形，渲染时用3角形？

家具装修玩法：
1. 像城市那样有街道，街道两边有房间，
2. 经济系统：鼓励玩家相互串门交流，增加金币；
3. 根据金币数量，选择户型图，建墙，购买家具，摆放家具；

游戏的玩家系统：
1. 玩家3D模型，多少自由度，皮肤可定制，玩家的高度设定对于家具装修体验很重要？
2. 前后左右移动，视野移动；
3. 放置、拆除物体，移动物体；

3D物理引擎
    1. Havok
    1. NovodeX
    1. [BulletPhysics](https://github.com/bulletphysics/bullet3), 开源
    1. ODE, 开源
    1. TOKAMAK, 开源
    1. Newton
    1. Simple Physics Engine, 国产
1. 碰撞检测

阅读文献：
1. [Explaining basic 3D theory](https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_on_the_web/Basic_theory)
2. 

倾斜摄影三维建模
全过程包括同名点提取、特征点匹配、三角网构建等的各种算法都得到了优化，但仍然无法解决以下类型物体重建：纹理贫乏（如水面、玻璃平面等大面积表面纹理单一的物体）运动目标（如移动中的车辆、随风摆动的树木等等）细杆和薄片物体（如路面标志牌、电线杆、电线等等）.
https://www.zhihu.com/question/279598635

3D-GAN

机器人寻路: 通过扫描光栅来反解建模的!

测绘&建模

微软Kinect或Asus Xtion传感器: 既能提供彩色图像又能提供密集深度图像的新型相机系统. 

深度学习生成3D模型:
http://www.bimant.com/blog/polygen-3d-models/
    * PolyGen : https://arxiv.org/abs/2002.10880 联合估计模型的面和顶点以直接生成网格
        * https://github.com/deepmind/deepmind-research/tree/master/polygen
        * 将 3D 模型表示为顶点和面的严格排序序列，而不是图像、体素或点云,应用基于注意力的序列建模方法来生成 3D 网格.
    * PointNet : https://arxiv.org/abs/1612.00593, 点云数据建模
    * 3D-R2N2: https://arxiv.org/abs/1604.00449, 体素,2D 卷积扩展到 3D，并自然地从 RGB 图像生成防水网格
    * Pixel2Mesh: https://arxiv.org/abs/1804.01654, 通过变形模板网格（通常是椭圆体）从单个图像预测 3D 模型的顶点和面。
    * 拓扑修改网络(TMN) ,https://arxiv.org/abs/1909.00321 ,引入两个新阶段在 Pixel2Mesh 上进行迭代：拓扑修改阶段用于修剪会增加模型重建误差的错误面，以及边界细化阶段以平滑由面修剪引入的锯齿状边界
    * https://arxiv.org/abs/1802.05384,https://arxiv.org/abs/1704.00710

    * 3D Machine Learning: https://github.com/timzhang642/3D-Machine-Learning
    * 3D模型表示: 体素、点云、网格、多视图图像集


深度学习3D合成:
https://cloud.tencent.com/developer/article/1779936


https://www.csdn.net/tags/MtzaYg5sNDk0NTUtYmxvZwO0O0OO0O0O.html
    (a) 点云；(b) 体素网格；(c) 三角网格；(d) 多视图表示


https://www.sohu.com/a/231225837_114778

https://blog.csdn.net/qq_39426225/article/details/101684526

产业：Bifrost,积木易搭,