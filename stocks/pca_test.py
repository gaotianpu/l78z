#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


df = pd.read_csv("predict/maxmin_today.csv",dtype={'stock_no':str})
# print(df[['mm_price','mm_VOTURNOVER','std_TCLOSE','std_VOTURNOVER']].head())

x = df[['mm_price','mm_VOTURNOVER','std_TCLOSE','std_VOTURNOVER']]
pca = PCA(n_components=2) #n_components 选择降维数量
pca = pca.fit(x) #开始拟合建模
ret_pca = pca.transform(x) #获得降维后数据

ret = pd.concat([df, pd.DataFrame(ret_pca,columns=['pca_x0','pca_x1'])], axis=1)


t = df[['mm_price','mm_VOTURNOVER']].to_numpy()
model = KMeans(n_clusters=16)
y_pred = model.fit_predict(x)
# print(y_pred)
ret = pd.concat([ret, pd.DataFrame(y_pred,columns=['cluster_idx'])], axis=1)

for index, row in ret.iterrows():
    # <div class="stock c0" style="left:10px;top:10px;"><a href="#">60000</a></div>
    # <div class="stock c0" style="left:1380px;bottom:810px;"><a href="#">60001</a></div>
    # 
    left_v = round((1380-10)*row['mm_price']/100+10)
    bottom_v = round((1380-10)*row['mm_VOTURNOVER']/100+10)
    
    div = '<div class="stock c%d" style="left:%dpx;top:%dpx;"><a href="http://stockpage.10jqka.com.cn/%s/" target="_blank"> * </a></div>'%(row['cluster_idx'],left_v,bottom_v,row['stock_no'])
    # print(row)
    print(div)
    # break 

ret.to_csv("predict/pca.csv", index=False)

plt.figure('Kmeans', facecolor='lightgray')
plt.gca().invert_yaxis() #原点位置默认左下，调整为左上
plt.title('Kmeans', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.tick_params(labelsize=10)
plt.scatter(t[:, 0], t[:, 1], s=80, c=y_pred, cmap='brg', label='Samples')
plt.legend()
plt.show()