1. 视频抽帧  extract_tool.py 
    1.1 人工抽帧  python extract_tool.py output/video_list_0522.txt manual
        快捷键 w\s\a\d,空格，仿照游戏设计
    1.2 随机抽帧  python extract_tool.py output/video_list_0522.txt random 20
        会存在较多不合适的样本？

2. 人工标注
    2.1 使用 labelImg, 注意快捷键的应用，PascalVOC/YOLO格式
    2.2 自己开发一套？ 借鉴vscode electronjs  +TODO
        效率优先
    2.3 宁缺毋滥！

3. 根据人工标注的数据扩充抽帧  +TODO
    python extended_sample.py 
    3.1  区域相似度计算、需找一个更好的算法 
    3.2  迭代训练，利用上一版本的训练模型，识别区域，生成标识数据、减少人工标注的工作量？
    3.3  每个视频应限制抽取的总量

4. PascalVOC格式转为YOLO格式
    4.1 python voc2yolo.py
        生成voc.names,labels,train.txt,validate.txt
        问题：YOLOv3在训练的时候会用到validate.txt吗？
    4.2 images,labels,cfg/voc.data 一次性设置好即可
        data/voc.names,cfg/yolov3-voc.cfg, 每次类目有调整需要变更，脚本自动化?
        
5. 训练日志分析、可视化
    python yolo_train_visual.py train.0529.log



其他问题：
1. YOLO训练数据曾么增广

鼠标事件定义：https://blog.51cto.com/devops2016/2084084