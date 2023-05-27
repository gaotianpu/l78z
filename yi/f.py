import os,sys

with open('yi.txt','r') as f:
    i = 0 
    tmp = ""
    for line in f:
        if line.strip():
            tmp = tmp + "|" + line.strip() 
            if i%10 == 9:
                print(tmp)
                tmp = "" 
                
            # print(i%10,line) 
            i = i + 1