#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys


root_dir = "output_0531"
JPEGImages_dir = os.path.join(root_dir,'JPEGImages')


def mv(no_index=1):
    target_dir = os.path.join(root_dir,'JPEGImages'+str(no_index))
    if not os.path.exists(target_dir):
        os.system('mkdir '+ target_dir)

    vid_list = []
    for i,fname in  enumerate(os.listdir(JPEGImages_dir)):
        vid = fname.split('_')[0]
        fx = fname.split('.')[0]
        ftype = fname.split('.')[1]
        if ftype=='png' and vid not in vid_list:
            vid_list.append(vid)
            mv1 = "mv %s %s" % (os.path.join(JPEGImages_dir,fname) , os.path.join(target_dir,fname))
            mv2 = "mv %s %s" % (os.path.join(JPEGImages_dir,fx+'.xml') , os.path.join(target_dir,fx+'.xml'))
            print(i,fname)
            os.system(mv1)
            os.system(mv2) 

mv(4)