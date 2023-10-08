

def bad(order_models):
    cnt = len(order_models)
    model_sets_li = [order_models]
    # 2
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
    
    print("len(model_sets_li):",len(model_sets_li))



import random
def the_sort_score(m):
    return random.random()

def good(): # n/(n+1)/2 = 28 次
    best_models = []
    
    current_best_score = 0
    for i in range(7):
        li_ = []
        for model in order_models:
            if model in best_models:
                continue
              
            m = best_models + [model] 
            score = the_sort_score(m)
            li_.append( [m,score] )
            
        best = max(li_, key = lambda x: x[1])
        if best[1] > current_best_score:
            best_models = best[0]
            current_best_score = best[1]
        else :
            print("???", best[0])
    
    print(best_models)


if __name__ == "__main__":
    order_models="point_pair_high,pair_11,pair_15,pair_16,point_4,point_5,point_high1".split(",")
    bad(order_models) # 120次
    good(order_models) # n/(n+1)/2 = 28 次
     