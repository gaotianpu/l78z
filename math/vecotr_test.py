#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两个向量距离
https://blog.csdn.net/qq_19707521/article/details/78479532
1.闵可夫斯基距离(Minkowski Distance)

2.欧氏距离(Euclidean Distance)

3.曼哈顿距离(Manhattan Distance)

4.切比雪夫距离(Chebyshev Distance)

5.夹角余弦(Cosine)

6.汉明距离(Hamming distance)

7.杰卡德相似系数(Jaccard similarity coefficient)

8.贝叶斯公式


"""
import numpy as np
from sklearn import preprocessing
import json 


def vector_distance(vector1, vector2):
    """两个向量的相似度"""
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    vector1 = np.around(vector1,decimals=6)
    vector2 = np.around(vector2,decimals=6)

    diff_vector = vector1-vector2

    op1 = np.sqrt(np.sum(np.square(diff_vector)))
    op2 = np.linalg.norm(diff_vector, ord=2)
    print("欧式距离：%s=%s" % (op1, op2))

    #
    op3=np.sum(np.abs(diff_vector))
    op4=np.linalg.norm(diff_vector,ord=1)
    print("曼哈顿距离: %s=%s" % (op3, op4))


    op5=np.abs(diff_vector).max()
    op6=np.linalg.norm(diff_vector,ord=np.inf)
    print("切比雪夫距离: %s=%s" % (op5, op6))

    # 夹角余弦
    op7 = np.dot(vector1, vector2) / \
        (np.linalg.norm(vector1,ord=2)*(np.linalg.norm(vector2,ord=2)))
    print("夹角余弦:%s" % op7)

     
    distance_square = np.sum(np.square(diff_vector))
    distance_square = 1.0 / (1.0 + np.exp(distance_square))
    #distance_square = tf.reduce_sum(tf.square(diff_embeddings), 1)
    # distance = distance_square
    print(distance_square)

    # op1=np.sqrt(np.sum(np.square(diff_embeddings)))
    # op2=np.linalg.norm(diff_embeddings)
    # print(op1)
    # print(op2)

def reg(v):
    v = np.array(v) 
    ret = v/np.sqrt(np.sum(np.square(v)) )
    # preprocessing.normalize(v.reshape(1,-1), norm='l2') #ret等价于此
    return ret 

def test_vreg():
    response = '{"error_code":52000,"result":[{"value":[-0.30988073348999026]},{"value":[-1.5599597692489625,-1.011673927307129,1.0856126546859742,0.18077200651168824,-3.2275900840759279,0.7282294034957886,3.7582128047943117,0.947734534740448,0.32910069823265078,2.0651044845581056,-1.6227161884307862,-0.39118945598602297,-1.8037644624710084,-0.697830855846405,0.2906566560268402,-1.1064578294754029]},{"value":[-1.3010985851287842,1.3756247758865357,-1.6669995784759522,-0.16876034438610078,3.766892194747925,2.4173502922058107,-0.5573937296867371,1.4918711185455323,-1.6414779424667359,-0.3798484206199646,3.11633038520813,-0.9098528623580933,-1.1963666677474976,3.424078941345215,-1.900799036026001,-2.4569966793060304]}]}'
    obj = json.loads(response)
    result = obj.get('result')
    score = result[0].get("value")[0]
    print("score:%s" % (score) )

    np.set_printoptions(precision=18)
    vector_query = np.array(result[1].get("value"))
    vector_doc = np.array(result[2].get("value"))
    
    q = preprocessing.normalize(vector_query.reshape(1,-1), norm='l2')
    doc = preprocessing.normalize(vector_doc.reshape(1,-1), norm='l2')
    print(vector_doc)
    print(doc)
    print(np.dot(q,doc.reshape(-1,1)))

    # print(vector_doc)
    # print(reg(vector_doc))  

if __name__ == "__main__":
    # vector_distance([3, 4], [4, 3])
    test_vreg()
    # reg([3, 4])

    # 0.5767171382904053
    # vector_distance([-2.9286251068115234, -0.2927297055721283, 0.8340327143669128, 1.2210958003997803, 0.8643559217453003, 1.5409256219863892, 3.0976874828338623, 3.1198439598083496, -1.3106415271759033, 1.4751449823379517, 1.0416207313537598, 0.02059176377952099, -1.5351008176803589, 2.8584554195404053, -0.5347332954406738, -1.8563042879104614],
    #                 [-1.3010984659194946, 1.3756250143051147, -1.6669995784759521, -0.16876070201396942, 3.766892671585083, 2.4173500537872314, -0.5573943853378296, 1.4918707609176636, -1.6414780616760254, -0.3798485994338989, 3.116330623626709, -0.9098528623580933, -1.1963660717010498, 3.424078941345215, -1.9007993936538696, -2.456996202468872])
