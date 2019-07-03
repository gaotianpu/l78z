#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging


class YOLOv3TrainVisualization:
    def __init__(self, train_log, result_dir='.'):
        self.train_log = train_log
        self.result_dir = result_dir

        self.loss_file = os.path.join(
            self.result_dir, "loss.%s" % (self.train_log))
        self.iou_file = os.path.join(
            self.result_dir, "iou.%s" % (self.train_log))

        self.loss_graph = os.path.join(self.result_dir, "loss")
        self.iou_graph = os.path.join(self.result_dir, "iou")

        self.extract_log()

    def extract_log(self):
        if not os.path.exists(self.loss_file):
            os.system("cat %s|grep images > %s" %
                      (self.train_log, self.loss_file))
        if not os.path.exists(self.iou_file):
            os.system('cat %s|grep IOU|grep -v nan|head > %s' %
                      (self.train_log, self.iou_file))

    def parse_loss_log(self, line_num=2000):
        result = pd.read_csv(self.loss_file,
                             skiprows=[x for x in range(line_num) if ((x % 10 != 9) | (
                                 x < 1000))],
                             error_bad_lines=False,
                             names=['loss', 'avg', 'rate', 'seconds', 'images'])
        result['loss'] = result['loss'].str.split(' ').str.get(1)
        result['avg'] = result['avg'].str.split(' ').str.get(1)
        result['rate'] = result['rate'].str.split(' ').str.get(1)
        result['seconds'] = result['seconds'].str.split(' ').str.get(1)
        result['images'] = result['images'].str.split(' ').str.get(1)

        result['loss'] = pd.to_numeric(result['loss'])
        result['avg'] = pd.to_numeric(result['avg'])
        result['rate'] = pd.to_numeric(result['rate'])
        result['seconds'] = pd.to_numeric(result['seconds'])
        result['images'] = pd.to_numeric(result['images'])
        return result

    def parse_iou_log(self, line_num=2000):
        result = pd.read_csv(self.iou_file,
                             skiprows=[x for x in range(line_num) if (
                                 x % 10 == 0 or x % 10 == 9)],
                             error_bad_lines=False,
                             names=['Region Avg IOU', 'Class', 'Obj', 'No Obj', 'Avg Recall', 'count'])
        result['Region Avg IOU'] = result['Region Avg IOU'].str.split(
            ': ').str.get(1)
        result['Class'] = result['Class'].str.split(': ').str.get(1)
        result['Obj'] = result['Obj'].str.split(': ').str.get(1)
        result['No Obj'] = result['No Obj'].str.split(': ').str.get(1)
        result['Avg Recall'] = result['Avg Recall'].str.split(': ').str.get(1)
        result['count'] = result['count'].str.split(': ').str.get(1)

        result['Region Avg IOU'] = pd.to_numeric(result['Region Avg IOU'])
        result['Class'] = pd.to_numeric(result['Class'])
        result['Obj'] = pd.to_numeric(result['Obj'])
        result['No Obj'] = pd.to_numeric(result['No Obj'])
        result['Avg Recall'] = pd.to_numeric(result['Avg Recall'])
        result['count'] = pd.to_numeric(result['count'])
        return result

    def save_loss_graph(self):
        pd_loss = self.parse_loss_log()

        fig = plt.figure(1, figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(pd_loss["avg"].values, label="Avg Loss", color="#ff7043")
        ax.legend(loc="best")
        ax.set_title("Avg Loss Curve")
        ax.set_xlabel("Batches")
        ax.set_ylabel("Avg Loss") 

        fig.savefig(self.loss_graph)

    def save_iou_graph(self):
        pd_loss = self.parse_iou_log()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(pd_loss['Region Avg IOU'].values, label='Region Avg IOU', color="#ff7043")
        # ax.plot(pd_loss['Class'].values,label='Class')
        # ax.plot(pd_loss['Obj'].values,label='Obj')
        # ax.plot(pd_loss['No Obj'].values,label='No Obj')
        # ax.plot(pd_loss['Avg Recall'].values,label='Avg Recall')
        # ax.plot(pd_loss['count'].values,label='count')
        ax.legend(loc='best')
        ax.set_title('The Region Avg IOU curves')
        ax.set_xlabel('batches')
        fig.savefig(self.iou_graph)


if __name__ == "__main__":
    train_log_file = sys.argv[1]
    obj = YOLOv3TrainVisualization(train_log_file,'./output') 
    obj.save_loss_graph()
    obj.save_iou_graph()

    # python yolo_train_visual.py train.0529.log
