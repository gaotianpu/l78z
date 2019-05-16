# -*- coding: utf-8 -*-
'''
借助createID.py生成训练集和验证集的图片名称列表trainImageId.txt和validateImageId.txt
'''
import os;
import shutil;
 
def listname(path,idtxtpath):
    filelist = os.listdir(path);  # 该文件夹下所有的文件（包括文件夹）
    filelist.sort()
    f = open(idtxtpath, 'w')
    for files in filelist:  # 遍历所有文件
        Olddir = os.path.join(path, files);  # 原来的文件路径
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
            continue
        f.write(files)
        f.write('\n')
    f.close()
 
savepath = os.getcwd()
imgidtxttrainpath = savepath+"/trainImageId.txt"
imgidtxtvalpath = savepath + "/validateImageId.txt"
listname(savepath + "/trainImage",imgidtxttrainpath)
listname(savepath + "/validateImage",imgidtxtvalpath)

print("trainImageId.txt && validateImageId.txt have been created!") 