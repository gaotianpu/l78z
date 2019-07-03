#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function
import os
import sys
import numpy as np
import cv2
import json
import time
import random


class ExtractTool(object):
    def __init__(self, video_list_file, root_dir='./output/'):
        '''初始化'''
        self.video_list_file = video_list_file
        self.root_dir = root_dir

        self.__create_dir()  # 创建约定的数据文件存放目录

        self.video_list = []
        self.video_list_len = 0
        self.get_video_list()

        self.__download_videos() #

        self.current_video_index = 0
        self.current_frame_index = 50
        self.current_cap = None

    def get_video_file(self, vid):
        return os.path.join(self.root_dir, "videos", "%s.mp4" % (vid))

    def get_frame_img_file(self, vid, frame_index, img_path='images'):
        str_file_name = "%s_%s.png" % (vid, frame_index)
        return os.path.join(self.root_dir, img_path, str_file_name)

    def get_video_list(self):
        '''获取视频列表，格式：video_id \t video_play_url'''
        if self.video_list:
            return self.video_list
        self.video_list = [line.strip().split("\t")
                           for line in open(self.video_list_file, 'r') if line]
        self.video_list_len = len(self.video_list)
        return self.video_list

    def __create_dir(self):
        '''按照约定，创建默认的文件夹'''
        #'images', 'Annotations', 'labels', 'Classes'
        path_list = ['videos', 'manual', 'manual_random']
        for path in path_list:
            my_path = os.path.join(self.root_dir, path)
            if not os.path.exists(my_path):
                os.makedirs(my_path)

    def __download_videos(self):
        '''下载视频数据'''
        for field in self.video_list:
            video_id = field[0]
            video_play_url = field[1]
            fname = os.path.join(self.root_dir, "videos", video_id+'.mp4')
            if not os.path.isfile(fname):
                os.system('wget %s -O %s' % (video_play_url, fname))

    def get_frame(self, video_index_change=1, frame_index_change=10):
        change_video = False
        if video_index_change != 0:
            tmp_index = self.current_video_index + video_index_change
            if tmp_index < 0:
                tmp_index = 0
            if tmp_index >= self.video_list_len:
                tmp_index = self.video_list_len-1
            if tmp_index != self.current_video_index:
                self.current_video_index = tmp_index
                self.current_frame_index = 50
                change_video = True

        if change_video or not self.current_cap:
            if self.current_cap:
                self.current_cap.release()
            video_file = self.get_video_file(
                self.video_list[self.current_video_index][0])
            self.current_cap = cv2.VideoCapture(video_file)
            self.frames_len = int(
                self.current_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # frame_index_change
        if frame_index_change != 0:
            tmp_index = self.current_frame_index + frame_index_change
            if tmp_index < 0:
                tmp_index = 0
            if tmp_index != self.current_frame_index:
                self.current_frame_index = tmp_index
            if tmp_index >= self.frames_len:
                tmp_index = self.frames_len-1
            if tmp_index != self.current_frame_index:
                self.current_frame_index = tmp_index

        self.current_cap.set(cv2.CAP_PROP_POS_FRAMES,
                             self.current_frame_index)  # 设置要获取的帧号
        ret, frame = self.current_cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        if ret:
            vid = self.video_list[self.current_video_index][0]
            tips = 'vid:%s,v_idx:%s/%s,f_idx:%s/%s' % (
                vid, self.current_video_index, self.video_list_len,
                self.current_frame_index, self.frames_len)
            print(tips)
            cv2.imshow(self.win_title, frame)

            # cv2.destroyWindow(self.win_title)
            # self.win_title = tips
            # cv2.namedWindow(self.win_title)

        return ret, frame

    def save_image(self):
        self.current_cap.set(cv2.CAP_PROP_POS_FRAMES,
                             self.current_frame_index)  # 设置要获取的帧号
        ret, frame = self.current_cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        if ret:
            vid = self.video_list[self.current_video_index][0]
            vid_file = self.get_frame_img_file(
                vid, self.current_frame_index, "manual")
            cv2.imwrite(vid_file, frame)
            tips = 'vid:%s,v_idx:%s/%s,f_idx:%s/%s' % (
                vid, self.current_video_index, self.video_list_len,
                self.current_frame_index, self.frames_len)
            #
            print('save:' + tips)

    def load_selected_vid(self):
        p = os.path.join(self.root_dir, 'manual')
        s = set()
        for f in os.listdir(p):
            x = f.split('_')
            if len(x) == 2:
                s.add(x[0])
        self.selected_vids = list(s)
        self.current_video_index = len(
            self.selected_vids) if self.selected_vids else 0
        # print(len(self.selected_vids))

    def extract_manual(self):
        '''人工视频抽帧'''
        self.win_title = 'frame'
        cv2.namedWindow(self.win_title)
        cv2.moveWindow(self.win_title, 10, 100)

        frame = np.zeros((500, 500, 3), np.uint8)
        self.load_selected_vid()
        self.get_frame(0, 0)

        while(True):
            cv2.imshow(self.win_title, frame)

            # 获取键盘事件
            flag = cv2.waitKey(1)
            if flag == 27:  # Esc，退出
                print('esc exit')
                break

            if flag == ord('w'):
                ret, frame = self.get_frame(0, 100)  # 下帧
            if flag == ord('s'):
                ret, frame = self.get_frame(0, -10)  # 上帧
            if flag == ord('a'):
                ret, frame = self.get_frame(-1, 0)  # 上一个视频
            if flag == ord('d'):
                ret, frame = self.get_frame(1, 0)  # 下一个视频

            if flag == 32:  # 32 空格键
                self.save_image()

            time.sleep(0.1)

        cv2.destroyAllWindows()

    def extract_random(self, need_count=20):
        """随机抽帧"""
        for i, field in enumerate(self.video_list):
            video_id = field[0]
            print(i, video_id)
            self.random_select_sample_single(video_id, need_count)

    def random_select_sample_single(self, video_id, need_count=20):
        """单个视频的随机抽帧"""
        video_file = self.get_video_file(video_id)
        cap = cv2.VideoCapture(video_file)  # 文件名及格式
        frame = np.zeros((500, 500, 3), np.uint8)
        frame_index = 0
        selected_count = 0
        while(True):
            # capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            r = random.randint(50, 500)
            if frame_index > 100 and frame_index % r == 0:
                if selected_count > need_count:
                    break
                frame_img_file = self.get_frame_img_file(
                    video_id, frame_index, 'manual_random')
                cv2.imwrite(frame_img_file, frame)
                selected_count = selected_count + 1

            frame_index = frame_index + 1
        cap.release()


def main():
    ''' python extract_tool.py vlist_0601.txt manual
        python extract_tool.py output/vlist_0601.txt random 30 
    '''
    video_list_file = sys.argv[1]
    extract_way = sys.argv[2]

    tool = ExtractTool(video_list_file)
    if extract_way == 'manual':
        tool.extract_manual()
    else:
        need_count = int(sys.argv[3])
        tool.extract_random(need_count)


def test():
    pass


if __name__ == "__main__":
    main()
