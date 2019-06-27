#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载视频，
    用户拖拽选定感兴趣区域，
    右键取消选中区域
    保存roi区域

鼠标事件定义：https://blog.51cto.com/devops2016/2084084
"""
import sys
import numpy as np
import cv2
import json
from skimage.measure import compare_ssim

MANUAL_DIR = 'output/manual'
IMAGES_DIR = 'output/JPEGImages'
CROP_IMAGES_DIR = 'output/CropImages'
LABELS_DIR = 'output/labels'

def convert(size, box):
    '''归一处理'''
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h) #中心点坐标，区域的宽高

def compare_image(grayA, imageB):
    '''比较2张图片的相似度'''
    # imageB = cv2.imread(path_image2)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    # print("SSIM: {}".format(score))
    return score


def run(video_id, video_file):
    '''对ROI区域抽帧裁剪'''
    #定义窗口
    win_title = 'frame'
    cv2.namedWindow(win_title)
    cv2.moveWindow(win_title, 10, 100)

    #fps ？
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    fps_params = cv2.CAP_PROP_FPS
    if int(major_ver) < 3:
        fps_params = cv2.cv.CV_CAP_PROP_FPS

    cap = cv2.VideoCapture(video_file)  # 文件名及格式
    fps = cap.get(fps_params)

    # 从人工选取的数据文件中加载要剪裁的区间
    sample_area = []
    selected_area = []
    manual_params = {}
    with open("%s/%s.json" % (MANUAL_DIR, video_id), 'r') as f:
        manual_params = json.load(f)
        selected_area = manual_params['selected']
        for i, area in enumerate(selected_area):
            manual_img = "%s/%s_%s.png" % (MANUAL_DIR, video_id, i)
            # print(manual_img)
            imageA = cv2.imread(manual_img)
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            sample_area.append(grayA)
    
    # class 
    all_classes = []
    with open('yolo_classes.txt', 'r') as f:
        all_classes = [line.strip() for line in f]

    frame = np.zeros((500, 500, 3), np.uint8)

    frame_index = 0
    while(True):
        #capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % (fps*1) == 0:  # 每间隔1s抽帧
            good_area = []
            for i, area in enumerate(selected_area):
                min_x = min(area[1][0], area[2][0])
                min_y = min(area[1][1], area[2][1])
                width = abs(area[1][0] - area[2][0])
                height = abs(area[1][1] - area[2][1])
                cropped_img = frame[min_y:min_y+height, min_x:min_x+width]
                # 计算选中区域与样例选中区域的相似度
                score = compare_image(sample_area[i], cropped_img)
                if score >= 0.3 and score < 0.98:
                    good_area.append(i)
                    print(i, frame_index, score)
                    sample_img = "%s/%s_%s_%s.png"%(CROP_IMAGES_DIR,video_id,i,frame_index) # media_id, selected_area_no,frame_index
                    cv2.imwrite(sample_img, cropped_img)

            if good_area:
                img_file = "%s/%s_%s.jpg" % (IMAGES_DIR,
                                                video_id, frame_index)
                cv2.imwrite(img_file, frame)


                label_file = "%s/%s_%s.json" % (LABELS_DIR,
                                                video_id, frame_index)
                yolo_file = "%s/%s_%s.yolo" % (LABELS_DIR,
                                                video_id, frame_index)  
                params = {'video_id':video_id,'selected':[],'class':manual_params['class'],'size':manual_params['size']}
                
                # 1. wh,是图片的，还是
                size = manual_params['size'] #图像本身的大小,[高,宽,depth]
                w = int(size[1])
                h = int(size[0])
                with open(yolo_file, 'w') as f: 
                    for a in good_area:
                        area_point = manual_params['selected'][a]
                        params['selected'].append(area_point)
                        min_x = min(area_point[1][0],area_point[2][0])
                        max_x = max(area_point[1][0],area_point[2][0])
                        min_y = min(area_point[1][1],area_point[2][1]) 
                        max_y = max(area_point[1][1],area_point[2][1])  
                        b = (float(min_x), float(max_x), float(min_y), float(max_y))
                        bb = convert((w,h), b)

                        class_name = manual_params['class'][a] 
                        class_index = all_classes.index(class_name)

                        f.write(str(class_index) + " " + " ".join([str(a) for a in bb]) + '\n')

                with open(label_file, 'w') as f: 
                    json.dump(params, f)  


        for area in selected_area:
            cv2.rectangle(frame, tuple(area[1]),
                          tuple(area[2]), (0, 255, 0), 1)

        # #display the resulting frame
        cv2.imshow(win_title, frame)
        if cv2.waitKey(25) & 0xFF == 27:  # 按esc键退出  # 0xFF == ord('q'):  # 按q键退出
            break

        frame_index = frame_index + 1

    #when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_id = sys.argv[1]
    video_file = "output/videos/%s.mp4" % (video_id)

    run(video_id, video_file)
