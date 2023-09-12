#!/usr/bin/python
import numpy as np
import pandas as pd 
import itertools

li =      [1,3,2,4]
li_name = ['a','b','c','d']
count = len(li)

def get_index(li,li_name):
    com_two = np.array(list(itertools.combinations(li, 2)))
    diff_two = np.diff(com_two)
    print(com_two,diff_two)
    print(diff_two[diff_two>1])
    print(np.argwhere(diff_two>1))
    # for t in itertools.combinations(li, 2):
    #     if abs(t[0]-t[1]) > 1.5:
    #         if t[0]>t[1]:
    #             print(t[0],t[1])
    #         else:
    #             print(t[1],t[0])
                
    
    # arr = np.array(list(itertools.combinations(li, 2)))
    # print(arr)
    
    #
    print("use numpy")
    count = len(li) 
    arr = np.expand_dims(li,1).repeat(count,1).T
    
    print(np.triu(arr,0)[1:,1:]) #ok 2,3,4;3,4;4;
    
    # # 1,2,3;1,2;1; ?
    # # print(np.fliplr(arr))
    print(np.tril(arr,0)[:-1,:-1])
    
    # # print(np.flip(np.flip(np.tril(arr,0)[:-1,:-1],axis=1),axis=0) ) 
    # # print(np.triu(np.flip(arr,axis=1),0)[1:,1:]) 
    # # print(np.triu(arr,0)[:-1,:-1]) #not ok. 
    
    
    
    # u4 = np.argwhere((np.triu(arr,0)[:-1,:-1] - np.triu(arr,0)[1:,1:]) > 0)
    # print("u4:", u4)
    # u4[:,1] = u4[:,1] + 1  
    
    # print(["%s,%s"%(li_name[item[1]],li_name[item[0]]) for item in u4.tolist()])
    # print(u4.tolist())
    # return u4

# arr = np.repeat(np.expand_dims(np.array(li),1),3,1).T # .reshape(3, 3).T
# arr = np.expand_dims(li,1).repeat(count,1).T
# print(arr)


# # L = np.tril(arr,-1)
# # print(L)

# u1 = np.triu(arr,0)[:-1,:-1]
# print(u1)

# u2 = np.triu(arr,0)[1:,1:]
# print(u2)

# u3 = u1 - u2
# print(u3)
# u4 = np.argwhere(u3>0)
# print(u4)

# u4 = np.argwhere((np.triu(arr,0)[:-1,:-1] - np.triu(arr,0)[1:,1:]) >0)
# u4[:,1] = u4[:,1] + 1 
# print(u4)

print("get_index:")
print(get_index(li,li_name))



# t = np.tril(arr)-np.tril(arr,-1)
# print(np.tril(arr))
# print(np.tril(arr,-1))
# print(t)

# arr = np.repeat(li,count).reshape(count, -1).T
# print(arr)

# arr = np.repeat(arr,3,1).T # .reshape(3, 3).T
# print(arr)




data_type = "train"
df = pd.read_csv("data/rnn_%s.txt"%(data_type),
        names="date,stock,high,low,high_label,low_label,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16".split(","),
        header=None, dtype={'stock':str,'high_label':int}, sep=";")

df.describe()