import numpy as np
import pandas as pd
import math
from sklearn.metrics import ndcg_score  

def compute_ndcg(df):
    ret = []
    date_groups = df.groupby(0)
    for date,data in date_groups: 
        data = data.sort_values(by=[2])
        data[4] = [math.ceil((i+1)/3) for i in range(20)]
        
        data = data.sort_values(by=[3],ascending=False)
        mean_3 = round(data[2].head(3).mean(),5)
        mean_all = round(data[2].mean(),5)
        
        y_true = np.expand_dims(data[4].to_numpy(),axis=0)
        y_predict = np.expand_dims(data[3].to_numpy(),axis=0)
        ndcg = round(ndcg_score(y_true,y_predict),3)
        ndcg_3 = round(ndcg_score(y_true,y_predict,k=3),3)
        # print(date,ndcg)
        ret.append([date,ndcg,ndcg_3,mean_3,mean_all])
        # break 
    return ret 

    # print(key,)
    # break 
# ValueError: Only ('multilabel-indicator', 'continuous-multioutput', 'multiclass-multioutput') formats are supported. Got continuous instead
    
# print(tmp)
# print(tmp.describe())

# with open("ndcg.txt","r") as f :
#     data = np.loadtxt(f,delimiter=";")
#     print(data)

df = pd.read_csv("ndcg.txt",sep=";", header=None)
ret = compute_ndcg(df)
for x in ret:
    print(x)
print(sum([x[1] for x in ret])/len(ret))
print(sum([x[2] for x in ret])/len(ret))
print(sum([x[3] for x in ret])/len(ret))
print(sum([x[4] for x in ret])/len(ret))