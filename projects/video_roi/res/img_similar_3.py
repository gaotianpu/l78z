import cv2
import matplotlib.pyplot as plt


def getss(list):				     #计算方差
    avg=sum(list)/len(list)          #计算平均值
    ss=0
    for l in list:                   #计算方差
        ss+=(l-avg)*(l-avg)/len(list)
    return ss


def getdiff(img):                     #获取每行像素平均值
    Sidelength=32                     #定义边长
    img=cv2.resize(img,(Sidelength,Sidelength),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    avglist=[]                        #avglist列表保存每行像素平均值
    for i in range(Sidelength):       #计算每行均值，保存到avglist列表
        avg=sum(gray[i])/len(gray[i])
        avglist.append(avg)
    return avglist

#读取测试图片
img1_path='test.mp4_0.png'
img2_path='test.mp4_auto_0_0.png'

img1=cv2.imread(img2_path)
diff1=getdiff(img1)
print('img1:',getss(diff1))

#读取测试图片
img11=cv2.imread(img2_path)
diff11=getdiff(img11)
print('img11:',getss(diff11))

x=range(32)

plt.figure("avg")
plt.plot(x,diff1,marker="*",label="$walk01$")
plt.plot(x,diff11,marker="*",label="$walk03$")
plt.title("avg")
plt.legend()
plt.show() 
