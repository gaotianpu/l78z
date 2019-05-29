#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import operator
import xml.etree.ElementTree as ET


class Voc2Yolo(object):
    def __init__(self, root_dir='./output/'):
        '''初始化'''
        self.root_dir = root_dir
        self.voc_dir = os.path.join(root_dir, 'Annotations')

        # 产出的
        self.yolo_dir = os.path.join(root_dir, 'labels')
        self.class_file = os.path.join(self.root_dir, 'voc.names') #class
        self.train_file = os.path.join(self.root_dir, 'train.txt')
        self.valid_file = os.path.join(self.root_dir, 'valid.txt')

        self.classes = []

    def get_classes(self):
        classes_dict = {}
        for f in os.listdir(self.voc_dir):
            if '.xml' != os.path.splitext(f)[1]:
                continue

            fname = os.path.join(self.voc_dir, f)
            root = ET.parse(open(fname)).getroot()

            for obj in root.iter('object'):
                class_name = obj.find('name').text
                classes_dict[class_name] = classes_dict.get(class_name, 0) + 1

        sorted_x = sorted(classes_dict.items(),
                          key=operator.itemgetter(1), reverse=True)

        # for x in sorted_x:
        #     print(x)
        # print('len(sorted_x):', len(sorted_x))
        self.classes = [x[0] for x in sorted_x]
        return self.classes

    def convert(self, size, box):
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
        return (x, y, w, h)  # 中心点坐标，区域的宽高

    def convert_annotation(self, image_id, xml_file):
        in_file = open(xml_file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')  # 图像本身的大小
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        out_file = open(os.path.join(
            self.yolo_dir, '%s.txt' % (image_id)), 'w')
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult) == 1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            # xmin,选取坐标
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
                xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = self.convert((w, h), b)
            out_file.write(str(cls_id) + " " +
                           " ".join([str(a) for a in bb]) + '\n')

    def process(self):
        self.get_classes()

        # 存储voc.names
        with open(os.path.join(self.root_dir, 'voc.names'), 'w') as classes_file:
            for class_name in self.classes:
                classes_file.write(class_name+"\n")

        train_file = open(self.train_file, 'w')
        val_file = open(self.valid_file, 'w')
        for i, f in enumerate(os.listdir(self.voc_dir)):
            tmp = os.path.splitext(f)
            file_id = tmp[0]
            file_type = tmp[1]

            if '.xml' != file_type:
                continue

            xml_file = os.path.join(self.voc_dir, f)
            self.convert_annotation(file_id, xml_file) 
            
            is_train = random.randint(0, 50) % 10 >= 3
            img_path = os.path.join("images", '%s.png' % (file_id))
            if is_train:
                train_file.write(img_path + "\n")
            else:
                val_file.write(img_path + "\n")


 
if __name__ == "__main__":
    obj = Voc2Yolo()
    obj.process() 
