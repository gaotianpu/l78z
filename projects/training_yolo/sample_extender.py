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
        
        print(filename.split('.')[0],class_name)

        return ret

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
        selected_index = 0
        while(True):
            # capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index<200:
                frame_index = frame_index + 1
                continue 

            if frame_index % (fps*2) == 0:  # 每间隔1s抽帧
                if selected_index>20:
                    break 
                
                # good_area = []
                # for roi_area in roi_area_list:
                #     cls_name = roi_area['class']
                #     if cls_name in [x['class'] for x in good_area]:
                #         continue

                #     # 截取感兴趣区域
                #     # area_points = (xmin,xmax,ymin,ymax,width,height)
                #     p = roi_area['area_points']
                #     cropped_img = frame[p[2]:p[2]+p[5], p[0]:p[0]+p[4]]

                    # # 比对,计算选中区域与样例选中区域的相似度
                    # score = self.compare_image(
                    #     roi_area['area_img'], cropped_img) 
                    # if score >= 0.3 and score < 0.98:
                    #     self.log_obj.info('vid=%s, frame_index=%s, cls_name=%s, score=%s' % (vid, frame_index, cls_name, score) )
                        
                        
                        # 保存ROI区域图片
                        # CLASS_PATH = "output/Classes"
                        # cls_path = os.path.join(CLASS_PATH, cls_name)
                        # roi_img = os.path.join(cls_path, '%s.%s_%s.png' % (
                        #     cls_name, vid, frame_index))
                        # cv2.imwrite(roi_img, cropped_img)

                        # good_area.append(roi_area)

                # if good_area:
                cls_li = [x['class'] for x in roi_area_list]
                tv = [x['class'] for x in roi_area_list if 'tv' in x['class'] and 'tvshow' not in x['class']]
                # print('cls_li:',cls_li)
                # print('left:', )
                
                # if len(good_area)==len(tv): 
                # # if len(good_area)>1:
                # # if len( [g for good_area if 'tv' in g['class'] and 'tvshow' not in g['class']] ) > 1:
                #     # print('googd_area', vid, len(good_area), frame_index)
                #     self.log_obj.info('only tv_logo,vid=%s,frame_index=%s' % (vid,frame_index))
                #     continue 

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

                for roi_area in roi_area_list:
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
                    break #only one

                # 保存xml文件
                # xml_file = "output/JPEGImages/%s_%s.xml" % (
                xml_file = "output/JPEGImages/%s_%s.xml" % (
                    vid, frame_index)
                mydata = ET.tostring(root, encoding="unicode")
                myfile = open(xml_file, "w")
                myfile.write(mydata)

                selected_index = selected_index + 1

            frame_index = frame_index + 1

        cap.release()
        cv2.destroyAllWindows()
    
    def process(self): 
        for video_id, video_url in self.video_list:
            select_area_list = []

            # 获取roi区域信息
            roi_area_list = self.get_manual_files(video_id) 

            # if roi_area_list:  
            #     self.process_single_video(video_id, roi_area_list)
            # else:
            #     pass
                # self.log_obj.info('%s has no roi_area_list' % (video_id) )


def main():
    se = SampleExtender('vlist_0601.txt')
    # se.get_manual_files('')
    # se.get_classes()
    se.process()
    pass


if __name__ == "__main__":
    main()
