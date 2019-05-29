#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import time
import random
import operator
import numpy as np
import cv2
import json
import xml.etree.ElementTree as ET
from skimage.measure import compare_ssim
import logging
import traceback


# 读取xml,扩充

class SampleExtender(object):
    def __init__(self, video_list_file, root_dir='./output/', extract_max_count=30):
        '''初始化'''
        self.video_list_file = video_list_file
        self.root_dir = root_dir
        self.extract_max_count = extract_max_count

        self.manual_root = os.path.join(root_dir, 'manual')
        self.manual_root = os.path.join(root_dir, 'Annotations')

        self.video_list = []
        self.video_list_len = 0
        self.get_video_list()

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
            self.log_obj.warn('load_single_xml,%s' % (path))
            # print(path)

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
            # CLASS_PATH = "output/Classes"
            # cls_path = os.path.join(CLASS_PATH, class_name)
            # if not os.path.exists(cls_path):
            #     os.makedirs(cls_path)
            # sample_img = os.path.join(cls_path, class_name + '.' + filename)
            # cv2.imwrite(sample_img, area_img)

            # area_img = None
            obj = {'file_name': filename, 'class': class_name,
                   'area_points': area_points, 'area_img': area_img}
            ret.append(obj)

        return ret


def main():
    se = SampleExtender('output/video_list_0522.txt')
    # se.get_manual_files('')
    se.get_classes()
    pass


if __name__ == "__main__":
    main()
