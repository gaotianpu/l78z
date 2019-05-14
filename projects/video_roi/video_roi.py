#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载图像，用户使用鼠标选择ROI区域，保存ROI区域

鼠标事件定义：https://blog.51cto.com/devops2016/2084084
"""
import numpy as np
import cv2 


def on_mouse(event, x, y, flags, param):
    param['current_pos'] = (x, y)
    if not param['is_draging'] and flags ==  cv2.EVENT_FLAG_LBUTTON:
        param['start_pos'] = param['current_pos']
        param['is_draging'] = True   

    if param['is_draging'] and flags !=  cv2.EVENT_FLAG_LBUTTON:
        param['end_pos'] = param['current_pos']
        param['is_draging'] = False 
        i = len(param['selected'])
        param['selected'].append( [i,param['start_pos'],param['end_pos']]  )   

        param['start_pos'] = None 
        param['end_pos'] = None  
    
    if  event == cv2.EVENT_RBUTTONDOWN:
        print("event", event)
        if param['selected']:
            param['selected'].pop() 

def run():
    cap = cv2.VideoCapture('test.mp4')  # 文件名及格式

    frame = np.zeros((500, 500, 3), np.uint8)
    cv2.namedWindow('frame')

    mouse_params = {'img':frame, 'is_draging': None, 'start_pos': None, 'current_pos': None,
        'end_pos': False, 'selected':[]}
    cv2.setMouseCallback('frame', on_mouse,mouse_params)

    while(True):
        #capture frame-by-frame
        ret, frame = cap.read()

        #our operation on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if mouse_params['start_pos'] and mouse_params['current_pos']:
            cv2.rectangle(frame, mouse_params['start_pos'], mouse_params['current_pos'], (0,255,0), 1)

        # if 0xFF == 27:  # 按esc键退出
        #     if mouse_params['selected']:
        #         mouse_params['selected'].pop() 

        for area in mouse_params['selected']: 
            cv2.rectangle(frame, area[1], area[2], (0,255,0), 1) 

        #display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) &  0xFF == 27 : #按esc键退出  # 0xFF == ord('q'):  # 按q键退出
            break

    #when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()  