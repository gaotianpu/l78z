# 平面几何 - 尺规作图工具
cwrac, construction with ruler and compasses

<img src="https://github.com/gaotianpu/l78z/blob/master/projects/cwrac/static/images/cwrac.png?raw=true" alt="平面几何 - 尺规作图工具" width="400">

DEMO: https://gaotianpu.github.io/cwrac/index.html

## 一、基本工具
1. 点 draw_point
    用户手动绘制的点带A,B,C这样的文本标记
    遍历所有手绘点、取最后一个，根据最后一个，设置文本。
2. 直线 draw_line
3. 圆 draw_arc
4. 相交点
 * 直线和直线   Line.line_intersect_point
 * 直线和圆 Circle.line_intersect_point
 * 圆和圆   Circle.circle_intersect_point


### 二、细节：高亮、磁吸、移动、去重等
1. 高亮
 * 在绘制过程中，当前绘制对象高亮；  
 * 当用户鼠标停留在线上、交点上，高亮。
2. 磁吸：在绘制过程中，当鼠标靠近点、靠近直线线或圆时，自动磁吸至该位置，防止用户误操作。  
 * 磁吸到点
 * 磁吸到线 【HOLD:暂时仅高亮】
    a. 当前点到线上最近点？
    b. 当前绘制对象和线的交点？ 
3. 去重：【HOLD】
 * 点去重 - 相同的位置画2个点(x,y距离)，磁吸功能已解决该问题
 * 线段去重 - 2条线(w,b的差值), 线段覆盖的问题
 * 圆去重 - 2个圆(x,y,radius)，应去重 
4. 误操作: 直线距离过短，圆半径过短，当前绘制对象不生效
5. 撤销/重做
6. 移动：
7. 缩放：【HOLD】 
8. 点击2下(等同拖拽效果)绘图？
    不同的工具对鼠标事件响应不一样？鼠标的点击、释放


## 三、高级工具 
高级工具用于加速作图效率，前提是懂得如何使用基本工具达到高级工具的效果，因此，在正式发布的版本中需用户逐步解锁。
另，高级工具依赖其他图形完备，例如：中垂线、过点垂线、平行线前提是已有一条直线；角平分线前提是至少有2条相交直线；
计算逻辑; 绘图&交互逻辑【TODO】  
1. 垂直平分线 (先确定平分点[x,y]，两直线垂直，它们的斜率相乘等于负一，a_new = -1/(a), 求b)
    垂线：两直线垂直，它们的斜率相乘等于负一, 斜率为0的情况,垂线斜率-1？
    斜率已知，线段中点坐标，根据中点坐标、求b，只画直线？
    用户操作：依次点击给定线段的两个端点
2. 过线外一点作垂线
    ([x,y]已知，两直线垂直，它们的斜率相乘等于负一)
    斜率已知，根据x,y点求b
3. 过point平行线 y=ax+b, a,x,y已知，求b=y-ax，容易
4. 圆规, 先选取2点确定半径，第3个点为圆心，容易
5. 角平分线 ？【TODO】
    三个点：顶点、两条边上各任意一点


## 四、其他: 【TODO】
1. UI优化：工具栏icons、undo/redo, canvas大小位置
* PC页面布局：canvas占据浏览器全部区域；工具栏浮动在左侧；undo/redo位于左上角；高级工具默认灰色需解锁？
* 移动版布局：工具栏底部、undo/redo位于工具栏上方，靠右；
* icons: 理想设计不用语言也能明白是干嘛的； 
2. 云存储，用户操作记录保存,上传，分享
    持久化：即使刷新页面，也不会清除已绘制的内容？【TODO】
    存储至服务器
3. 操作步骤回放，录制视频？
4. 关卡设计


## 五、渠道&推广
基本功能搞完后，可以先发布一版？云功能是否必须？
1. 独立网页
2. 微信小程序
    https://developers.weixin.qq.com/miniprogram/dev/framework/
3. 头条小程序
    https://developer.toutiao.com/docs/
4. Windows/Mac/Linux App
    https://electronjs.org/
5. Android App
6. iOS    
7. 手百小程序
    https://smartprogram.baidu.com/docs/develop/tutorial/codedir/



## 六、总结
1. 程序采用MVC结构，model-数据-用户的绘图对象，view-render,将用户绘制对象展现出来，control-鼠标事件，将事件操作结果转换为数据m
* M: 点、线、圆类，有一个数组，用于记录每个步骤干嘛,
* C: 事件响应，鼠标的点击、释放, event_binding,
* V: 渲染 render