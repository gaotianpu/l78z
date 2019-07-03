import os 

vid_count = {}
for i,f in  enumerate(os.listdir('output/Annotations')): 
    x = f.split('_')
    vid = x[0]
    vid_count[vid] = vid_count.get(vid,0) + 1 


for k,v in vid_count.items():
    print(k,v)