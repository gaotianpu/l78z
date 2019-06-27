YOLOv3训练自己的数据

关键点在于：弄清YOLOv3数据集的组织形式，需要按照这个生成数据。

一. 训练环境
    1. YOLOv3编译：OpenCV+GPU
        git clone https://github.com/pjreddie/darknet 
        cd darknet
        vim ... 

        1.1 查看显卡
            lspci | grep -i vga  
            lspci | grep -i nvidia
            nvidia-smi 
            watch -n 0.2 nvidia-smi 
        1.2  
         
         

    2. 下载预训练权重(只包含卷积层)
        wget https://pjreddie.com/media/files/darknet53.conv.74 


二. 数据准备
    0. 因为YOLOv3提供了一个将VOC数据集转成自己训练的数据集格式的工具:scripts/voc_label.py,
        因此，训练数据只需要按VOC数据集的格式准备就好, 再执行voc_label.py即可
        另一种思路，直接生成YOLO格式的？YOLO的格式更简洁些。
    1. VOC2012
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/  
        研究下文件组织结构
        Annotations:用于存放与图片对应的XML文件
        JPEGImages: 存放所有的图片，png是否可以？
    2. 对自己的图片数据打标,按照YOLOv3的方式准备数据集
        人工选取： l78z/projects/video_roi/video_roi_manual.py
        自动保存： l78z/projects/video_roi/video_roi_crop.py
        其他：labelImg: https://github.com/tzutalin/labelImg#macos
            很多教程是基于次标注工具产出数据，再转成yolo训练数据
            该工具支持直接生成YOLO格式，
            基于cvui搞个工具？
                ctrl+t , 打开新视频
                左键选择，右键取消
    3. 使用 scripts/voc_label.py 生成 


三、 修改YOLOv3的配置文件
    1. data/voc.names  #修改成自己的分类
    2. cfg/voc.data
       classes = N       #（N为自己的分类数量，如有10类不同的对象，N = 10）
       train = train.txt    # 训练集完整路径列表
       valid = valid.txt   # 测试集完整路径列表
       names = data/voc.names    # 类别文件
       backup = backup     #（训练结果保存在darknet/backup/目录下）
      存放图片的路径默认是images, 存放标注数据labels.
    3. cfg/yolov3-voc.cfg
        classes = N （N为自己的分类数）
        filters = 3*(classes+1+4)，  修改每一个[yolo]层（一共有3处）之前的filters.
        (可选) 修改训练的最大迭代次数， max_batches = N
    4. cfg/yolov3-voc.cfg
        在正式的训练机器上跑
        batch=64  #每batch个样本更新一次参数。 
        subdivisions=16  #如果内存不够大，将batch分割为subdivisions个子batch，每个子batch的大小为batch/subdivisions。



四、 YOLOv3训练&测试
    watch -n 0.2 nvidia-smi 

    训练：
    ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74 -gpus 0,1,3 
    nohup ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74 -gpus 1,6,7 > train.log 2>&1  &
    
    nohup ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74  -gpus 1,6,7 > train.log 2>&1  &

    # 中断后继续训练
    提示：
    Loaded: 0.000806 seconds 
    Region 106 Avg IOU: 0.208178, Class: 0.458221, Obj: 0.428369, No Obj: 0.439916, .5R: 0.062500, .75R: 0.000000,  count: 16
    Region(82,94,106) ?  82预测区域最大，106预测区域最小
        Avg IOU 表征在当前subdivision内的图片的平均IOU，代表预测的矩形框和真实目标的交集与并集之比；
        Class表征标注物体分类的正确率，期望该值趋近于1；
        Obj越接近1越好；
        No Obj期望该值越来越小，但不为零；
        count是所有的当前subdivision图片中包含正样本的图片的数量。 

        是否可以断点接着训练？ backup?

    测试：
        一次测一张图片： ./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup output/manual_random/9973032745594506200_666.png
        一次测多张图片： 
        val ?

五、 生产环境部署问题
https://blog.csdn.net/phinoo/article/details/83009061

    https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
    https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
    http://emaraic.com/blog/yolov3-custom-object-detector
    https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV
    https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
    https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
    https://blog.csdn.net/haoqimao_hard/article/details/82081285
    https://blogs.oracle.com/meena/object-detection-using-opencv-yolo


六、题外话，说说视频标注工具的设计
    1. 读取video_list.txt，创建目录文件，下载视频，存放至约定的文件夹，文件价格结构：
        video_list.txt,  video_id \t video_download_url
        Videos/  用于存放下载的视频
        Manual/  用于存放用户选择的图片、区域、区域裁剪图
        JPEGImages/   存放自动生成的图片
        Annotations/  用于图片对应的选取信息
        class.txt    所有分类汇总
        class/          选取区域的图像 ？
            class1/file1.jpg
    2. 自动定位至还没有标注数据的，显示标注进度。 
    3. 快捷键设定：按游戏的设定w前进帧，s后退帧，d下一个，a上一个（上下切换视频，左右调节帧，方向键位于右侧，影响鼠标操作），上下切换时，自动保存。
    4. 拖拽选中区域，要有x/y辅助线，右键取消上一次选中， 
    5. 选中结束，弹窗，类名，设置类名时，除了文字外有对应的缩略图  
    6. 保存VOC格式， 全部结束后，自动生成YOLO格式。
    7. 可根据图片名搜索
    8. class缩略图，每个class对应的图片
    7. 默认画圈
    8. 鼠标移动区域，编辑？
    9. 聚类算法，视频中差异最大的n帧？
    10.images和xml默认放在同一个目录
    11. 有哪些类名，类名下有多少张图片，实时统计？防止出现错误
    12. 多人协同？
    13. 开放的类集合，如何设定标准？

参考文档：
https://blog.csdn.net/dcrmg/article/details/81296520
https://www.jianshu.com/p/f4518fe04da1
https://www.jianshu.com/p/91eafe0f3719
https://www.cnblogs.com/pprp/p/9525508.html

https://blog.csdn.net/gusui7202/article/details/83781719
https://www.jianshu.com/p/7b7420890639
https://blog.csdn.net/csdn_zhishui/article/details/85397380