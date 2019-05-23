#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
import cv2
import json
import xml.etree.ElementTree as ET
from skimage.measure import compare_ssim
import logging
import traceback
import time
import random

logging.basicConfig(filename='log/video_label.log',  level=logging.DEBUG,
                    format='%(funcName)s:%(lineno)d:%(levelname)s:%(asctime)s:%(message)s')


def on_mouse(event, x, y, flags, param):
    param['current_pos'] = (x, y)
    print(x, y)


def on_mouse_manual_select(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.imwrite(param['frame_img_file'], param['img'])
        print(param['frame_img_file'])
        time.sleep(0.5)

        # img2 = param['img'].copy()
        # cv2.circle(img2, (x,y),20, (0, 255, 0), 1)
        # cv2.imshow(param['win_title'], img2)


class VideoLabelTool(object):
    def __init__(self, video_list_file, log_obj=logging, root_dir='./output/'):
        '''初始化'''
        self.video_list_file = video_list_file
        self.root_dir = root_dir

        self.create_dir()  # 创建约定的数据文件存放目录

        self.video_list = []
        self.get_video_list()

        self.classes_set = set()  # 存储分类

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
        return self.video_list

    def create_dir(self):
        '''按照约定，创建默认的文件夹'''
        path_list = ['videos', 'manual', 'manual_random',
                     'images', 'Annotations', 'labels', 'Classes']
        for path in path_list:
            my_path = os.path.join(self.root_dir, path)
            if not os.path.exists(my_path):
                os.makedirs(my_path)

    def download_videos(self):
        '''下载视频数据'''
        for field in self.video_list:
            video_id = field[0]
            video_play_url = field[1]
            fname = os.path.join(self.root_dir, "videos", video_id+'.mp4')
            if not os.path.isfile(fname):
                os.system('wget %s -O %s' % (video_play_url, fname))

    def manual_select_single(self, video_id):
        '''人工选取抽帧-单个视频的处理'''
        video_file = self.get_video_file(video_id)
        cap = cv2.VideoCapture(video_file)  # 文件名及格式

        frame = np.zeros((500, 500, 3), np.uint8)
        win_title = 'frame:' + video_id
        cv2.namedWindow(win_title)
        cv2.moveWindow(win_title, 10, 100)

        mouse_params = {'video_id': video_id, 'img': frame,
                        'frame_index': 0, 'win_title': win_title}
        cv2.setMouseCallback(win_title, on_mouse_manual_select, mouse_params)

        frame_index = 0
        while(True):
            # capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            mouse_params['img'] = frame
            mouse_params['frame_img_file'] = self.get_frame_img_file(
                video_id, frame_index, 'manual')

            cv2.imshow(win_title, frame)

            if cv2.waitKey(25) & 0xFF == 27:  # 按esc键退出  # 0xFF == ord('q'):  # 按q键退出
                break

            frame_index = frame_index + 1

        # when everything done , release the capture
        cap.release()
        cv2.destroyAllWindows()

    def manual_select(self):
        '''人工选取抽帧'''
        # 已经标注过的,从manual文件中获取已有的列表？
        manual_selected = []
        current_video_id = ''

        for field in self.video_list:
            video_id = field[0]
            video_play_url = field[1]
            if video_id in manual_selected:
                continue
            self.manual_select_single(video_id)

    def random_select_sample_single(self, video_id):
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
                if selected_count > 30:
                    break
                frame_img_file = self.get_frame_img_file(
                    video_id, frame_index, 'manual_random')
                cv2.imwrite(frame_img_file, frame)
                selected_count = selected_count + 1

            frame_index = frame_index + 1
        cap.release()

    def random_select_sample(self):
        """随机抽帧"""
        for i, field in enumerate(self.video_list):
            video_id = field[0]
            print(i, video_id)
            self.random_select_sample_single(video_id)

    def get_manual_files(self, video_id, file_type='.xml'):
        '''根据nid，获取选中的图片'''
        flist = []
        select_area_list = []
        for f in os.listdir("output/manual"):
            if video_id in f and file_type == os.path.splitext(f)[1]:
                print(f)
                flist.append(f)

                # for fname in flist:
                area = self.load_single_xml(
                    os.path.join("output/manual", f))
                select_area_list.extend(area)

        return select_area_list

    def load_single_xml(self, manual_xml):
        tree = ET.parse(open(manual_xml))
        root = tree.getroot()
        size = root.find('size')
        size.find('width').text
        size.find('height').text

        filename = root.find('filename').text
        path = root.find('path').text
        try:
            full_img = cv2.imread(path)
        except:
            print(path)

        ret = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            class_name = obj.find('name').text

            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymin = int(xmlbox.find('ymin').text)
            ymax = int(xmlbox.find('ymax').text)

            # 裁剪选取区域保存至
            width = int(abs(xmin - xmax))
            height = int(abs(ymin - ymax))
            area_img = full_img[ymin:ymin+height, xmin:xmin+width]

            area_points = (xmin, xmax, ymin, ymax, width, height)

            # 存储选取区域的图片？
            CLASS_PATH = "output/Classes"
            cls_path = os.path.join(CLASS_PATH, class_name)
            if not os.path.exists(cls_path):
                os.makedirs(cls_path)
            sample_img = os.path.join(cls_path, class_name + '.' + filename)
            cv2.imwrite(sample_img, area_img)

            # area_img = None
            obj = {'file_name': filename, 'class': class_name,
                   'area_points': area_points, 'area_img': area_img}
            ret.append(obj)

        return ret

    def compare_image(self, imageA, imageB):
        '''比较2张图片的相似度'''
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        # print("SSIM: {}".format(score))
        return score

    def process_single_video(self, vid, roi_area_list):
        # 定义窗口
        win_title = 'frame'
        cv2.namedWindow(win_title)
        cv2.moveWindow(win_title, 10, 100)

        # fps ？
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        fps_params = cv2.CAP_PROP_FPS
        if int(major_ver) < 3:
            fps_params = cv2.cv.CV_CAP_PROP_FPS

        # 加载视频
        cap = cv2.VideoCapture(self.get_video_file(vid))  # 文件名及格式
        fps = cap.get(fps_params)

        # # 准备感兴趣区域
        # for roi_area in roi_area_list:
        #     roi_area['area_img_gray'] = cv2.cvtColor(
        #         roi_area['area_img'], cv2.COLOR_BGR2GRAY)
        #     pass

        frame = np.zeros((500, 500, 3), np.uint8)

        # 抽帧
        frame_index = 0
        while(True):
            # capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % (fps*2) == 0:  # 每间隔1s抽帧
                good_area = []
                for roi_area in roi_area_list:
                    cls_name = roi_area['class']
                    if cls_name in [x['class'] for x in good_area]:
                        continue

                    # 截取感兴趣区域
                    # area_points = (xmin,xmax,ymin,ymax,width,height)
                    p = roi_area['area_points']
                    cropped_img = frame[p[2]:p[2]+p[5], p[0]:p[0]+p[4]]

                    # 比对,计算选中区域与样例选中区域的相似度
                    score = self.compare_image(
                        roi_area['area_img'], cropped_img)

                    if score >= 0.3 and score < 0.98:
                        print(vid, frame_index, cls_name, score)
                        # 保存ROI区域图片
                        CLASS_PATH = "output/Classes"
                        cls_path = os.path.join(CLASS_PATH, cls_name)
                        roi_img = os.path.join(cls_path, '%s.%s_%s.png' % (
                            cls_name, vid, frame_index))
                        cv2.imwrite(roi_img, cropped_img)

                        good_area.append(roi_area)

                if good_area:
                    if len(good_area) > 1:
                        print('googd_area', vid, len(good_area), frame_index)

                    # 保存帧图片
                    str_file_name = "%s_%s.png" % (vid, frame_index)
                    frame_img_file = "output/JPEGImages/%s" % (str_file_name)
                    cv2.imwrite(frame_img_file, frame)

                    # create the file structure
                    root = ET.Element('annotation')
                    ET.SubElement(root, 'folder').text = 'JPEGImages'
                    ET.SubElement(root, 'filename').text = str_file_name
                    ET.SubElement(root, 'path').text = frame_img_file
                    source = ET.SubElement(root, 'source')
                    ET.SubElement(source, 'database').text = 'Unknown'

                    size_value = frame.shape  # 高rows、宽colums、the pixels
                    size = ET.SubElement(root, 'size')
                    ET.SubElement(size, 'width').text = str(size_value[1])
                    ET.SubElement(size, 'height').text = str(size_value[0])
                    ET.SubElement(size, 'depth').text = str(size_value[2])
                    ET.SubElement(root, 'segmented').text = '0'

                    for roi_area in good_area:
                        object_node = ET.SubElement(root, 'object')
                        ET.SubElement(
                            object_node, 'name').text = roi_area['class']
                        ET.SubElement(object_node, 'pose').text = 'Unspecified'
                        ET.SubElement(object_node, 'truncated').text = '0'
                        ET.SubElement(object_node, 'difficult').text = '0'

                        # area_points = (xmin,xmax,ymin,ymax,width,height)
                        point = roi_area['area_points']
                        bndbox = ET.SubElement(object_node, 'bndbox')
                        ET.SubElement(bndbox, 'xmin').text = str(point[0])
                        ET.SubElement(bndbox, 'ymin').text = str(point[2])
                        ET.SubElement(bndbox, 'xmax').text = str(point[1])
                        ET.SubElement(bndbox, 'ymax').text = str(point[3])

                    # 保存xml文件
                    xml_file = "output/Annotations/%s_%s.xml" % (
                        vid, frame_index)
                    mydata = ET.tostring(root, encoding="unicode")
                    myfile = open(xml_file, "w")
                    myfile.write(mydata)

            frame_index = frame_index + 1

        cap.release()
        cv2.destroyAllWindows()

    def test(self, vid):
        select_area_list = []

        # 获取roi区域信息
        roi_area_list = self.get_manual_files(vid)
        self.process_single_video(vid, roi_area_list)

        # # self.process_single_video(vid, roi_area_list)
        # for a in roi_area_list:
        #     a['area_img']=None
        #     print(a)

    def auto_generate(self):
        for video_id, video_url in self.video_list:
            select_area_list = []

            # 获取roi区域信息
            roi_area_list = self.get_manual_files(video_id)
            # print(roi_area_list)

            self.process_single_video(video_id, roi_area_list)


if __name__ == "__main__":
    tool = VideoLabelTool('output/video_list_0522.txt')
    # 0. 下载视频
    # tool.download_videos()

    # 1. 生成标注样本
    #   1.1. 随机抽帧作为标注样本
    # tool.random_select_sample_single('352027908165947368')
    tool.random_select_sample()

    #   1.2. 人工挑选标注样本？
    # tool.manual_select_single('5347276433421392089')
    # tool.manual_select()

    # 2. 如何在这个工具里实现roi区域选取？

    # 3. 根据标注样本，抽帧增广数据

    # print(tool.get_video_list())
    # tool.auto_generate()
    # tool.test('10890285572784318038')
