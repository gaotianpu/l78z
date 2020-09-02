#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np 

# 1. 生成数据
# 2. 预测模型
# 3. 损失函数

class LogisticRegression:
    def __init__(self,feathers_count,iter_num,learning_rate=0.001):
        self.weights = np.ones((feathers_count+1,1))
        self.iter_num = iter_num
        self.learning_rate = learning_rate
    
    def fc(self,x):
        """回归线"""
        X = np.c_[np.array(x), np.ones(len(x))]
        return X.dot(self.weights) 
    
    def sigmod(self,x):
        """sigmod"""
        # RuntimeWarning: overflow encountered in exp
        return 1.0 / (1.0 + np.exp(-x))

    def predict_score(self,x):
        """预测得分"""
        ret = self.sigmod(self.fc(x))
        return ret 
    
    def predict(self,x):
        """实际预测值: 1 or 0"""
        ret = self.predict_score(x)
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # return 1 if ret > 0.5 else 0 
        return [1 if  item > 0.5 else 0  for item in ret]

    def loss(self,x,y):
        """损失函数:交叉熵损失函数"""
        pre_y = self.predict_score(x)
        tmp = y * np.log(pre_y) + (1-y)*np.log(1-pre_y)
        # print("tmp:", tmp,y.shape)
        # print(tmp.reshape(1,-1))
        # t1 =  np.sum(tmp) / y.shape[0]
        t2 = tmp.mean()
        # assert(t1,t2)
        # print(t1,t2)
        # return t2
        return tmp 
         
    
    def optimize(self):
        """获取梯度，优化weights"""
        Err = self.loss(x,y) 
        Gradient = self.X.T.dot(Err).T/self.data_len
        self.weights = self.weights + self.learning_rate*Gradient
        pass 

    def fit(self,x,y): 
        self.x = x
        self.y = y 
        for i in range(self.iter_num):
            self.optimize() 

    
def unitTest():
    x = np.array([[1,2,3],[4,5,6]])
    y = np.array([[1],[1]])
    print(x.shape)

    lr = LogisticRegression(x.shape[1],50)
    print(lr.weights)

    print("## sigmod: ======")
    print(lr.sigmod(500))
    print(lr.sigmod(0))
    print(lr.sigmod(-500))
    print(lr.sigmod(np.array([1000,0,-1000])))

    print("## fc: ======")
    print(lr.fc(x))

    print("## predict: ======")
    print(lr.sigmod(np.array([7,16]) )) 
    print(lr.predict_score(x))
    print(lr.predict(x))

    print("## loss: ======")
    print(lr.loss(x,y))


if __name__ == "__main__":
    unitTest()