import os 

# nid='5347276433421392089,6084963048620346873'
# frame_index_start = 6700,46873


tmp_img ='output/tmp_img'

x='5347276433421392089,6700;6084963048620346873,46873;9105036619135791599,8850'
for vidfra in x.split(';'):
    tmp = vidfra.split(',')
    nid = tmp[0]
    frame_index_start = int(tmp[1]) 

    for f in os.listdir('output/tmp_img'):
        if nid in f:
            frame_index = int(f.split('.')[0].split('_')[1])
            if frame_index>frame_index_start:
                print("rm -f " + os.path.join(tmp_img,f) )
                os.system("rm -f " + os.path.join('output/tmp_img',f) )
                print("rm -f " + os.path.join('output/Annotations',f) )
                os.system("rm -f " + os.path.join('output/Annotations',f) )

        