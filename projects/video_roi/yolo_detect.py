#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import pdb
import numpy as np
import cv2
import xml.etree.ElementTree as ET

sys.path.append(os.path.join(os.getcwd(), 'python/'))
import darknet as dn



class YoloValidate(object):
    def __init__(self, folder, weights="yolov3-voc_10000.weights", gpu=-1,prob=0.8):
        '''初始化'''
        if gpu>=0:
            dn.set_gpu(gpu)
        self.folder = folder
        self.prob_threshold = prob
        self.net = dn.load_net("cfg/yolov3-voc.cfg".encode("utf-8"),
                               weights.encode("utf-8"), 0)
        self.meta = dn.load_meta("cfg/voc.data".encode("utf-8"))

    def save_voc_data(self, img_file, detect_result):
        img_path = os.path.join(self.folder, img_file)
        # 参考： https://gist.github.com/goodhamgupta/7ca514458d24af980669b8b1c8bcdafd
        # create the file structure
        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = self.folder
        ET.SubElement(root, 'filename').text = img_file
        ET.SubElement(root, 'path').text = img_path
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'

        img = cv2.imread(img_path)
        size_value = img.shape  # 高rows、宽colums、the pixels
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(size_value[1])
        ET.SubElement(size, 'height').text = str(size_value[0])
        ET.SubElement(size, 'depth').text = str(size_value[2])
        ET.SubElement(root, 'segmented').text = '0' 
        
        for item in detect_result:
            class_name = item[0].decode("utf-8")  
            prob = item[1]
            if prob<self.prob_threshold:
                continue 
            
            # print(img_file,class_name,prob)
            (center_x, center_y, bbox_width, bbox_height) = item[2]

            object_node = ET.SubElement(root, 'object')
            ET.SubElement(
                object_node, 'name').text = class_name
            ET.SubElement(object_node, 'pose').text = 'Unspecified'
            ET.SubElement(object_node, 'truncated').text = '0'
            ET.SubElement(object_node, 'difficult').text = '0'

            bndbox = ET.SubElement(object_node, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(
                int(center_x - (bbox_width / 2)))
            ET.SubElement(bndbox, 'ymin').text = str(
                int(center_y - (bbox_height / 2)))
            ET.SubElement(bndbox, 'xmax').text = str(
                int(center_x + (bbox_width / 2)))
            ET.SubElement(bndbox, 'ymax').text = str(
                int(center_y + (bbox_height / 2)))

        # 保存xml文件
        xml_file = os.path.join(
            self.folder, os.path.splitext(img_file)[0]+'.xml')
        mydata = ET.tostring(root, encoding="unicode")
        myfile = open(xml_file, "w")
        myfile.write(mydata) 

        return True 


    def detect_image(self, img_file):
        img_path = os.path.join(self.folder, img_file)

        ret = dn.detect(self.net, self.meta, img_path.encode("utf-8"), 0.4, 0.4)

        #detect class
        classes = []
        for item in ret:
            prob = item[1]
            if prob<self.prob_threshold:
                continue   
            class_name = item[0].decode("utf-8") 
            if class_name not in classes:
                classes.append(class_name)
            print(class_name,prob)
        classes.sort()

        #[(b'ad_bojuelvpai', 0.8590198159217834, (408.9674987792969, 306.0235595703125, 62.77572250366211, 45.864112854003906))]
        self.save_voc_data(img_file,ret)
        return classes

    
    def detect_images(self):
        for i,fname in  enumerate(os.listdir(self.folder)):
            # print(os.path.splitext(fname)[1])
            if '.png' != os.path.splitext(fname)[1]:
                continue  
            classes = self.detect_image(fname) 
            print('vid', i,fname,','.join(classes))

    def detect_videos(self,vid,video_file):
        pass 

    def load_voc_data(self, xml_file):
        # 人工标注，该图片有哪些ROI区域
        classes = []
        root = ET.parse(open(xml_file)).getroot()
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            class_name = obj.find('name').text
            if class_name not in classes:
                classes.append(class_name)
        return classes





if __name__ == "__main__":
    obj = YoloValidate('output/test','backup/yolov3-voc.backup')
    obj.detect_images()


    # run it
    # nohup python -u detect.py > detect.log 2>&1  &

    # obj.detect_image('16412594432904555506_10.png') 

# with open('tvshow0525_val.txt') as f:
#     for i,line in enumerate(f):
#         fname = line.strip()
#         img_id = fname.split("/")[1].split(".")[0]
#         print(img_id)
    # r = dn.detect(net, meta, fname.encode("utf-8"))
    # print(r)
