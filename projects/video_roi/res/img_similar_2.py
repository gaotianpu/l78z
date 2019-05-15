import cv2

#均值哈希算法
def aHash(img):
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    ahash_str=''
    for i in range(8):                  #遍历累加求像素和
        for j in range(8):
            s=s+gray[i,j]
    avg=s/64                            #求平均灰度
    for i in range(8):                  #灰度大于平均值为1相反为0生成图片的hash值
        for j in range(8):
            if  gray[i,j]>avg:
                ahash_str=ahash_str+'1'
            else:
                ahash_str=ahash_str+'0'
    return ahash_str


#差值感知算法
def dHash(img):
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dhash_str=''
    for i in range(8):                #每行前一个像素大于后一个像素为1，相反为0，生成哈希
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                dhash_str = dhash_str+'1'
            else:
                dhash_str = dhash_str+'0'
    return dhash_str

def cmpHash(hash1,hash2):              #Hash值对比
    n=0
    if len(hash1)!=len(hash2):         #hash长度不同则返回-1代表传参出错
        return -1
    for i in range(len(hash1)):        #遍历判断
        if hash1[i]!=hash2[i]:         #不相等则n计数+1，n最终为相似度
            n=n+1
    return n


if __name__ == '__main__':
    img1_path='test.mp4_0.png'
    img2_path='test.mp4_auto_0_0.png'

    img1=cv2.imread(img1_path)
    img2=cv2.imread(img2_path)
    hash1= aHash(img1)
    hash2= aHash(img2)
    print(hash1)
    print(hash2)
    n=cmpHash(hash1,hash2)
    print('均值哈希算法相似度：',n)


    hash1= dHash(img1)
    hash2= dHash(img2)
    print(hash1)
    print(hash2)
    n=cmpHash(hash1,hash2)
    print('差值哈希算法相似度：',n) 