#!/usr/bin/env python
# -*- coding: utf-8 -*-

total_count = 1949615  
group_count = 243702 #total_count分成8分

#  cat all_data.txt | awk -F ',' '{print $3}' | sort -n -r > max_mean.txt
# wc -l max_mean.txt
with open('data/max_mean.txt','r') as f:
    li = []
    for i,line in enumerate(f):
        if i>=group_count and i%group_count==0:
            print(i,line.strip())
            li.append(line.strip())
    print(",".join(li))


    