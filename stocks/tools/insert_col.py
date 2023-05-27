#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

def process(line):
    parts = line.strip().split(',')
    high = float(parts[2])
    low  = float(parts[3])

    max_high,min_high,max_low,min_low = 0.818,-0.6,0.668,-0.6
    label_high = round(16*(high-min_high)/(max_high-min_high))
    label_low = round(16*(low-min_high)/(max_low-min_low)) 

    parts.insert(4,str(label_low))
    parts.insert(4,str(label_high))
    print(",".join(parts))

if __name__ == "__main__":
    for i,line in enumerate(sys.stdin):
        process(line)