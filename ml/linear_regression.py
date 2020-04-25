#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np


def generate_train_data():
    """y = ax + b + random(0,1)"""
    l = []
    a = 9.0
    b = 4.0
    for x in range(10):
        y = a*x + b + (random.random()-0.5)
        l.append((y, x))
        # print(y,x)
    # print(l)
    return l


def generate_train_data_2():
    """y = ax + b + random(0,1)"""
    l = []
    a = 9.0
    b = 4.0
    c = 3.1
    for x in range(10):
        y = a*x + b*(x+2) + c 
        l.append((y, x, x+2))
        # print(y,x,x+2)
    # print(l)
    return l

class LinearRegression():
    def __init__(self, train_li, learning_rate=0.01, num_iter=1000):
        self.train_li = train_li
        self.data_len = len(train_li)
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.m_a = 0.0
        self.m_b = 0.0

    def predict(self, x):
        y = self.m_a * x + self.m_b
        return y

    def loss(self):
        s = 0
        for i, val in enumerate(self.train_li):
            y = val[0]
            x = val[1]
            pre_y = self.predict(x)
            s = s + (pre_y - y) * (pre_y - y)
        return s/self.data_len

    def set_weight(self, m_a, m_b):
        self.m_a = m_a
        self.m_b = m_b

    def compute_gradient_1(self):
        a_gradient = 0
        b_gradient = 0
        for i, val in enumerate(self.train_li):
            y = val[0]
            x = val[1]
            pred_y = self.predict(x)
            # a_gradient += -(1 / self.data_len) * (y - pred_y)*x
            # b_gradient += -(1 / self.data_len) * (y - pred_y)*1.0

            a_gradient -= (y - pred_y)*x / self.data_len
            b_gradient -= (y - pred_y)*1.0 / self.data_len

        self.set_weight(self.m_a - self.learning_rate*a_gradient,
                        self.m_b - self.learning_rate*b_gradient)

    def compute_gradient(self):
        """对比compute_gradient"""
        a_gradient = 0
        b_gradient = 0
        for i, val in enumerate(self.train_li):
            y = val[0]
            x = val[1]
            pred_y = self.predict(x)
            a_gradient -= (y - pred_y)*x
            # 等价，注意 y-pred_y 和前面的符号，求导影响这个？实际值-预测值这个顺序不能变？
            b_gradient -= (y - pred_y)
            # a_gradient += (pred_y - y)*x
            # b_gradient += (pred_y - y)

        a_gradient = a_gradient/self.data_len
        b_gradient = b_gradient/self.data_len

        self.set_weight(self.m_a - self.learning_rate*a_gradient,
                        self.m_b - self.learning_rate*b_gradient)

    def fit(self):
        print(self.m_a, self.m_b)
        for i in range(self.num_iter):
            self.compute_gradient()
        print(self.m_a, self.m_b)


def martix_test():
    X = np.array([[1, 2], [3, 4], [7, 8]])
    W = np.array([5, 6])
    Y = np.array([[15], [35], [82]])
    print("shape:", X.shape, Y.shape, W.shape)

    print("===点乘，损失函数均方误差======")
    print("X*W:", X*W)
    print("X.dot(W):", X.dot(W))
    print("Y.T:", Y.T)
    print("X.dot(W)-Y.T:", X.dot(W)-Y.T)
    print("np.square", np.square(X.dot(W)-Y.T))
    print("np.sum", np.sum(np.square(X.dot(W)-Y.T)))
    print("np.loss", np.sum(np.square(X.dot(W)-Y.T))/Y.shape[0])
    print("np.loss1", np.mean(np.square(X.dot(W)-Y.T)))
    print("np.loss1.1", np.square(X.dot(W)-Y.T).mean())
    # print("np.loss1.1", (X.dot(W)-Y.T).square().mean()) #error

    print("===权重初始化======")
    W_test = np.zeros((1, 2))
    print(W_test)
    print(X.shape[1])
    W_test_1 = np.zeros((1, X.shape[1]+1))
    print(W_test_1)

    print("====X扩展一列值=1===")
    xli = [[1, 2], [3, 4], [7, 8]]
    x = np.array(xli)
    print(x)
    # x = np.column_stack((x,1)) #error
    ones = np.ones(x.shape[0])
    print(np.c_[x, ones])  # 以下4种方式等价，使用最简单的
    print(np.column_stack((x, ones)))
    print(np.insert(x, x.shape[0]-1, values=ones, axis=1))
    print(np.c_[np.array(xli), np.ones(x.shape[0])])


class LinearRegressionWithMatrix():
    def __init__(self, X_li, Y_li, learning_rate=0.001, num_iter=10000):
        # X.shape=(sample_len,feature_len+1)  需要扩展一列bias=1
        self.X = np.c_[np.array(X_li), np.ones(len(X_li))]
        self.Y = np.array([Y_li]).T     # shape=(sample_len,1)
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.data_len = len(self.X)
        self.W = np.zeros((1, self.X.shape[1]))
        print(self.X.shape,self.Y.shape,self.W.shape)
        print(self.X)
        print(self.Y)

    def predict(self, X):
        return X.dot(self.W.T)

    def loss(self):
        return np.square(self.X.dot(self.W)-self.Y.T).mean()

    def optimize(self):
        pred_Y = self.predict(self.X)
        Err = self.Y - pred_Y
        Gradient = self.X.T.dot(Err).T/self.data_len
        self.W = self.W + self.learning_rate*Gradient
        
        # print(self.X.shape, pred_Y.shape, self.Y.shape, Err.shape, self.W.shape)

        # self.W = self.W + self.learning_rate * \
        #     self.X.T.dot(self.Y - self.predict(self.X)).T/self.data_len 
        
        # print(Err) 
        # print("Gradient:",Gradient)
        # print(np.mean(self.X.T.dot(Err))) #error
        # Gradient = np.mean(((self.Y - pred_Y).T)*self.X)
        # Gradient = np.mean((self.X*(().T)))

    def fit(self):
        # self.num_iter = 1
        for i in range(self.num_iter):
            self.optimize()
        print("w:", self.W)


def adjust_weight_by_manual(train_li):
    linearr = LinearRegression(train_li)

    linearr.set_weight(0, 0)
    print(linearr.loss())

    linearr.set_weight(4, 3)
    print(linearr.loss())

    linearr.set_weight(7, 3)
    print(linearr.loss())

    linearr.set_weight(9, 4)
    print(linearr.loss())


def adjust_weight_auto(train_li):
    linearr = LinearRegression(train_li)
    linearr.fit()


def adjust_auto_martix(train_li):
    X_li = [val[1:] for val in train_li]
    Y_li = [val[0] for val in train_li]
    linearr = LinearRegressionWithMatrix(X_li, Y_li)
    linearr.fit()


train_li = generate_train_data()
# adjust_weight_by_manual(train_li)

# print("============")

# adjust_weight_auto(train_li)
print("============")

# martix_test()

train_li = generate_train_data_2()
adjust_auto_martix(train_li)
