#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载视频，
    用户拖拽选定感兴趣区域，
    右键取消选中区域
    保存roi区域

鼠标事件定义：https://blog.51cto.com/devops2016/2084084
"""
import os 
import sys
import numpy as np
import cv2
import json
import time 


class VideoLabelTool2(object):
    def __init__(self, video_list_file, root_dir='./output/'):
        '''初始化'''
        self.video_list_file = video_list_file
        self.root_dir = root_dir

        self.create_dir()  # 创建约定的数据文件存放目录

        self.video_list = []
        self.video_list_len = 0 
        self.get_video_list()

        self.current_video_index = 0
        self.current_frame_index = 10
        self.current_cap = None 

    def get_video_file(self, vid):
        return os.path.join(self.root_dir, "videos", "%s.mp4" % (vid))

    def get_frame_img_file(self, vid, frame_index, img_path='images'):
        str_file_name = "%s_%s.png" % (vid, frame_index)
        # frame_img_file = "output/JPEGImages/%s" % (str_file_name)

        return os.path.join(self.root_dir, img_path, str_file_name)

    def get_video_list(self):
        '''获取视频列表，格式：video_id \t video_play_url'''
        if self.video_list:
            return self.video_list
        self.video_list = [line.strip().split("\t")
                           for line in open(self.video_list_file, 'r') if line]
        self.video_list_len = len(self.video_list)
        return self.video_list

    def create_dir(self):
        '''按照约定，创建默认的文件夹'''
        path_list = ['videos', 'manual', 'manual_random',
                     'images', 'Annotations', 'labels', 'Classes']
        for path in path_list:
            my_path = os.path.join(self.root_dir, path)
            if not os.path.exists(my_path):
                os.makedirs(my_path)
    
    def get_frame(self,video_index_change=1,frame_index_change=10):
        change_video = False 
        if video_index_change!=0:
            tmp_index = self.current_video_index + video_index_change
            if tmp_index<0:
                tmp_index = 0
            if tmp_index>=self.video_list_len:
                tmp_index = self.video_list_len-1
            if tmp_index!=self.current_video_index:
                self.current_video_index = tmp_index 
                self.current_frame_index = 10 
                change_video = True 

        if change_video or not self.current_cap:
            if self.current_cap:
                self.current_cap.release()
            video_file = self.get_video_file(self.video_list[self.current_video_index][0])
            self.current_cap = cv2.VideoCapture(video_file)
            self.frames_len = int(self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # frame_index_change 
        if frame_index_change!=0:
            tmp_index = self.current_frame_index + frame_index_change
            if tmp_index<0:
                tmp_index = 0
            if tmp_index!=self.current_frame_index:
                self.current_frame_index = tmp_index 
            if tmp_index>=self.frames_len:
                tmp_index = self.frames_len-1
            if tmp_index!=self.current_frame_index:
                self.current_frame_index = tmp_index

        self.current_cap.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame_index)  #设置要获取的帧号
        ret, frame =self.current_cap.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        if ret:
            # cv2.destroyWindow(self.win_title) 
            vid = self.video_list[self.current_video_index][0]
            # self.win_title = 'vid:%s,v_idx:%s/%s,f_idx:%s/%s' % (vid,self.current_video_index,self.video_list_len,self.current_frame_index,self.frames_len)
            # cv2.namedWindow(self.win_title)
            info = 'vid:%s,v_idx:%s/%s,f_idx:%s/%s' % (vid,self.current_video_index,self.video_list_len,self.current_frame_index,self.frames_len)
            print(info)
            cv2.imshow(self.win_title, frame) 
        return ret, frame 


    def run(self):
        self.win_title = 'frame'
        cv2.namedWindow(self.win_title)
        cv2.moveWindow(self.win_title, 10, 100) 

        frame = np.zeros((500, 500, 3), np.uint8)
        while(True):
            cv2.imshow(self.win_title, frame) 

            #获取键盘事件
            flag = cv2.waitKey(1)
            if flag == 27:  #Esc，退出
                print('esc exit')
                break  
             
            if flag == ord('w'):
                ret, frame = self.get_frame(0,100)  #下帧
            if flag == ord('s'):
                ret, frame = self.get_frame(0,-10) #上帧
            if flag == ord('a'):
                ret, frame = self.get_frame(-1,0) #上一个视频 
            if flag == ord('d'):
                ret, frame = self.get_frame(1,0)  #下一个视频 

            if flag == 32:
                print('space') 
            
            time.sleep(0.1)
        
        cv2.destroyAllWindows()
            


if __name__ == "__main__":
    tool = VideoLabelTool2('output/video_list_0522.txt') 
    tool.run()

    # video_id = sys.argv[1]
    # video_file = "output/videos/%s.mp4" % (video_id)
    # run(video_id, video_file)
