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


def on_mouse(event, x, y, flags, param):
    param['current_pos'] = (x, y)

    # 开始拖拽，记录起始坐标
    if not param['is_draging'] and flags == cv2.EVENT_FLAG_LBUTTON:
        param['start_pos'] = param['current_pos']
        param['is_draging'] = True

    # 拖拽中，实时汇出矩形框
    if param['is_draging'] and flags == cv2.EVENT_FLAG_LBUTTON:
        img2 = param['img'].copy()
        cv2.rectangle(img2, param['start_pos'],
                      param['current_pos'], (0, 255, 0), 1)
        cv2.imshow('frame', img2)

    # 拖拽结束
    if param['is_draging'] and flags != cv2.EVENT_FLAG_LBUTTON:  # : 
        param['is_draging'] = False

        param['end_pos'] = param['current_pos']
        
        #判断选择区域是否有效
        if param['start_pos'][0]!=param['end_pos'][0] and param['start_pos'][1]!=param['end_pos'][1]:  
            i = len(param['selected'])
            param['selected'].append([i, param['start_pos'], param['end_pos']])

            #裁剪选取区域保存至
            min_x = min(param['start_pos'][0],param['end_pos'][0])     
            min_y = min(param['start_pos'][1],param['end_pos'][1])
            width = abs(param['start_pos'][0] - param['end_pos'][0])
            height = abs(param['start_pos'][1] -param['end_pos'][1])
            cropped_img = param['img'][min_y:min_y+height, min_x:min_x+width] 
            sample_img = "%s_%s.png"%(param['video_id'],i)
            cv2.imwrite(sample_img, cropped_img) 
        

        param['start_pos'] = None
        param['end_pos'] = None

    # 单击右键,撤销上一个圈圈
    if param['selected'] and event == cv2.EVENT_RBUTTONDOWN:
        param['selected'].pop()


def run(video_id,video_file): 
    cap = cv2.VideoCapture(video_file)  # 文件名及格式

    frame = np.zeros((500, 500, 3), np.uint8)
    win_title = 'frame'
    cv2.namedWindow(win_title)
    cv2.moveWindow(win_title, 10, 100)

    mouse_params = {'video_id':video_id, 'img': frame, 'is_draging': None, 'start_pos': None, 'current_pos': None,
                    'end_pos': False, 'selected': []}
    cv2.setMouseCallback('frame', on_mouse, mouse_params)

    while(True):
        #capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break 

        mouse_params['img'] = frame

        # 暂停、再选取？
        if cv2.waitKey(25) & 0xFF == ord('p'):
            cv2.waitKey(0)

        if mouse_params['start_pos'] and mouse_params['current_pos']:
            cv2.rectangle(
                frame, mouse_params['start_pos'], mouse_params['current_pos'], (0, 255, 0), 1)

        for area in mouse_params['selected']:
            cv2.rectangle(frame, area[1], area[2], (0, 255, 0), 1)

        #display the resulting frame
        cv2.imshow(win_title, frame)
        if cv2.waitKey(25) & 0xFF == 27:  # 按esc键退出  # 0xFF == ord('q'):  # 按q键退出 
            # 将用户选中区域坐标数据存储起来
            with open(video_id + ".json", 'w') as f:
                mouse_params['img'] = None 
                json.dump(mouse_params, f) 
            break

    #when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_id = sys.argv[1]
    video_file = sys.argv[2]
    run(video_id,video_file)
