import numpy as np

def test():
    order_models="point_pair_high,pair_11,pair_15,pair_16,point_4,point_5,point_high1".split(",")
    cnt = len(order_models)

    # 2
    model_sets_li = [order_models]
    for i in range(cnt-1):
        for j in range(i+1,cnt):
            model_sets_li.append([order_models[i],order_models[j]])

    # 3
    for i in range(cnt-2):
        for j in range(i+1,cnt-1):
            for k in range(j+1,cnt):
                model_sets_li.append([order_models[i],order_models[j],order_models[k]])
    # 4
    for i in range(cnt-3):
        for j in range(i+1,cnt-2):
            for k in range(j+1,cnt-1):
                for l in range(k+1,cnt):
                    model_sets_li.append([order_models[i],order_models[j],order_models[k],order_models[l]])

    #5 
    for b in range(cnt-4):
        for i in range(b+1,cnt-3):
            for j in range(i+1,cnt-2):
                for k in range(j+1,cnt-1):
                    for l in range(k+1,cnt):
                        model_sets_li.append([order_models[b],order_models[i],order_models[j],order_models[k],order_models[l]])
    #6
    for a in range(cnt-5):
        for b in range(a+1,cnt-4):
            for i in range(b+1,cnt-3):
                for j in range(i+1,cnt-2):
                    for k in range(j+1,cnt-1):
                        for l in range(k+1,cnt):
                            model_sets_li.append([order_models[a],order_models[b],order_models[i],order_models[j],order_models[k],order_models[l]])

    for m in model_sets_li:
        mli = sorted(m)
        print(",".join(mli))

def zscore(x,mean,std):
    return round((x-mean)/std,4)

def minmax(x,min,max):
    return round((x-min)/(max-min),4)

def test2():
    data = [1,2,33,4,5,6,7,8,19]
    mean = np.mean(data)
    std = np.std(data)
    max_v = np.max(data)
    min_v = np.min(data)
    
    zscores = [zscore(x,mean,std) for x in data]
    max_vz = np.max(zscores)
    min_vz = np.min(zscores)
    print(zscores)
    
    for i in range(len(data)):
        print(minmax(data[i],min_v,max_v),minmax(zscores[i],min_vz,max_vz) )
        
        

test2() 