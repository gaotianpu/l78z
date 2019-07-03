搞一个开源的尺规作图工具? 不必急着写代码，先把Euclidea打通关！
空间形状、测量、逻辑、推理

参考：
1. Euclidea
    特点，像玩游戏一样设置关卡目标，通过若干关卡后可以解锁一个工具，鼓励多样性解题
    https://www.euclidea.xyz/en/game/
    问了2个小学一年的孩子，1个2年纪的孩子，如何激发孩子们的兴趣还是很关键。
2. GeoGebra 
    工具很强大，据说是开源的，但没找到开源代码。
    https://www.geogebra.org/geometry
3. 几何画板 The Geometer's Sketchpad， 简称为GSP
    传统的商业软件~
    http://www.jihehuaban.com.cn/
    
要实现的功能列表：
基础功能：
1. 基于 Web canvas 画点、直线、圆
    https://juejin.im/post/5c1da8e16fb9a049ef2690bc
    http://taobaofed.org/blog/2015/12/21/webgl-handbook/

    https://developer.mozilla.org/zh-CN/docs/Mozilla/Add-ons/Code_snippets/Canvas
    https://joshondesign.com/p/books/canvasdeepdive/toc.html
    http://bucephalus.org/text/CanvasHandbook/CanvasHandbook.html 
2. 计算相交点？
3. 用户交互，鼠标、键盘事件？
4. check 给定题目是否正确
5. 支持操作步骤的回放，编辑，录屏
UI：
6. 基于electron封装打包成一个跨平台的工具
    https://github.com/electron/electron
6. 功能多起来之后，代码如何组织
7. 专业的UI设计 
进阶：
8. 基于WebGL 3D 搞 ?


目标：
第一阶段：主要操作完成
第二阶段：app打包发布
第三阶段：网站托管


有用的阅读资料：
1. 需要了解一些动画制作的知识，用户画图时能更流畅
https://developer.mozilla.org/zh-CN/docs/Web/API/Canvas_API/Tutorial/Basic_animations
https://developer.mozilla.org/zh-CN/docs/Web/API/Canvas_API/Tutorial/Advanced_animations
添加鼠标控制
https://developer.mozilla.org/zh-CN/docs/Games/Anatomy #游戏中的主循环
https://joshondesign.com/p/books/canvasdeepdive/chapter04.html

https://developer.mozilla.org/zh-CN/docs/Games/Tutorials/2D_Breakout_game_pure_JavaScript?

3D?
https://developer.mozilla.org/zh-CN/docs/Games/Techniques/3D_on_the_web/Basic_theory




其他：
HTML5 Web Audio API？

http://www.w3school.com.cn/tags/html_ref_canvas.asp
https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Basic_usage


Web canvas API: 2D
https://developer.mozilla.org/zh-CN/docs/Web/API/Canvas_API/Tutorial/Drawing_shapes
如果你绘制出来的图像是扭曲的, 尝试用width和height属性为<canvas>明确规定宽高，而不是使用CSS。
左上角（坐标为（0,0））
lineCap = type， 设置线条末端样式。
lineJoin = type， 设定线条与线条间接合处的样式
miterLimit = value, 限制当两条线相交时交接处最大长度；所谓交接处长度（斜接长度）是指线条交接处内角顶点到外角顶点的长度。
如果你并不需要与用户互动，你可以使用setInterval()方法，它就可以定期执行指定代码。如果我们需要做一个游戏，我们可以使用键盘或者鼠标事件配合上setTimeout()方法来实现。通过设置事件监听，我们可以捕捉用户的交互，并执行相应的动作。

setInterval，定期执行 
setTimeout， 定时执行，需要与用户交互的情况，使用键盘或者鼠标事件配合上setTimeout()方法来实现
requestAnimationFrame(callback)

https://developer.mozilla.org/zh-CN/docs/Games/Tutorials/2D_Breakout_game_pure_JavaScript





https://developer.mozilla.org/zh-CN/docs/Games/Tutorials/2D_Breakout_game_pure_JavaScript

高分辨率的时间，

WebGL API: 3D
https://developer.mozilla.org/zh-CN/docs/Web/API/WebGL_API/Tutorial/Adding_2D_content_to_a_WebGL_context
着色器,使用OpenGL ES Shading Language(GLSL)编写的程序，
