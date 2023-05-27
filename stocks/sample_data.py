#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import random

#每天采样100组(?待定)，每组数量不确定，需要记录group_count

def load_data(model_type="all",data_type="train",label_idx=4):
    sep = "," if model_type=="all" else ";"
    with open('data/%s_%s.txt' % (model_type,data_type),'r') as f :
        last_date = "0"
        day_data = {}
        for i,line in enumerate(f):
            parts = line.strip().split(sep)
            label = parts[label_idx]
            if parts[0] != last_date or last_date=="0":
                if last_date!="0":
                    # print(last_date)
                    yield last_date,day_data
                last_date =  parts[0]
                day_data = {} 

            li = day_data.get(label,[])
            li.append(line.strip())
            day_data[label] = li            
            
data_type = "train"

def process(model_type,data_type,group_count=500):
    rows = load_data(model_type,data_type)

    f_train = open('data/%s_sample_%s.txt' % (model_type,data_type),'w')
    f_group = open('data/%s_sample_group_%s.txt' % (model_type,data_type),'w')

    for i,(last_date,day_data) in enumerate(rows):
        for sample_group in range(group_count):
            li = []
            for label in range(14):
                lines = day_data.get(str(label),[])
                if lines:
                    line = random.choice(lines)
                    li.append(line)
                    # print(line)
            f_train.write("\n".join(li)+"\n")
            f_group.write(str(len(li))+"\n")

# python sample_data.py all train  &
# python sample_data.py all validate & 
# python sample_data.py rnn train  &
# python sample_data.py rnn validate & 
if __name__ == "__main__":
    model_type = sys.argv[1]
    data_type = sys.argv[2]
    group_count = int(sys.argv[3])
    print(model_type,data_type)
    assert data_type in ["train","validate"] , "train,validate"
    process(model_type,data_type,group_count)


