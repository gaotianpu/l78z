import numpy as np
import pandas as pd
import math
from sklearn.metrics import ndcg_score,dcg_score  
import torch
import torch.nn.functional as F
import torch.nn as nn

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

# df = pd.read_csv("ndcg.txt",sep=";", header=None)
# ret = compute_ndcg(df)
# for x in ret:
#     print(x)
# print(sum([x[1] for x in ret])/len(ret))
# print(sum([x[2] for x in ret])/len(ret))
# print(sum([x[3] for x in ret])/len(ret))
# print(sum([x[4] for x in ret])/len(ret))


def ndcg_f(scores):
    """
    compute the NDCG value based on the given score
    :param scores: a score list of documents
    :return:  NDCG value
    """
    return dcg_f(scores)/idcg_f(scores)


class LambdaRankLoss(nn.Module):
    def dcg_f(self,scores):
        v = 0
        for i in range(len(scores)):
            v += (np.power(2, scores[i]) - 1) / np.log2(i+2)  # i+2 is because i starts from 0
        return v
    
    def single_dcg(self,scores, i, j):
        return (np.power(2, scores[i]) - 1) / np.log2(j + 2)

    def idcg_f(self,scores): 
        best_scores = sorted(scores)[::-1]
        return self.dcg_f(best_scores) 

    def ideal_dcg(self,scores):
        scores = [score for score in sorted(scores)[::-1]]
        return self.dcg_f(scores)

    def get_good_pair(self,true_scores):
        pairs = []
        count = len(true_scores)
        for i in range(0,count-1):
            for j in range(i+1,count): 
                if true_scores[i] > true_scores[j]:
                    pairs.append((i,j))
        return pairs

    # def forward(
    #     self, true_scores: torch.Tensor, predicted_scores: torch.Tensor
    # ) -> torch.Tensor:
    def forward(self,true_scores,predicted_scores):
        idcg = self.ideal_dcg(true_scores) #[ideal_dcg(scores) for scores in true_scores]
        print(idcg)
        
        good_ij_pairs = self.get_good_pair(true_scores)
        # true_scores, predicted_scores, good_ij_pairs, idcg, query_key = args
        num_docs = len(true_scores)
        sorted_indexes = np.argsort(predicted_scores)[::-1]
        rev_indexes = np.argsort(sorted_indexes)
        true_scores = true_scores[sorted_indexes]
        predicted_scores = predicted_scores[sorted_indexes]
        print(good_ij_pairs)
        
        lambdas = np.zeros(num_docs)
        # w = np.zeros(num_docs)

        single_dcgs = {}
        for i,j in good_ij_pairs:
            if (i,i) not in single_dcgs:
                single_dcgs[(i,i)] = self.single_dcg(true_scores, i, i)
            single_dcgs[(i,j)] = self.single_dcg(true_scores, i, j)
            if (j,j) not in single_dcgs:
                single_dcgs[(j,j)] = self.single_dcg(true_scores, j, j)
            single_dcgs[(j,i)] = self.single_dcg(true_scores, j, i)
        
        print(single_dcgs)
        for i,j in good_ij_pairs:
            z_ndcg = abs(single_dcgs[(i,j)] - single_dcgs[(i,i)] + single_dcgs[(j,i)] - single_dcgs[(j,j)]) / idcg
            rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
            # rho_complement = 1.0 - rho
            lambda_val = z_ndcg * rho
            lambdas[i] += lambda_val
            lambdas[j] -= lambda_val 
        
        return lambdas

if __name__ == "__main__":
    # test()
    
    true_scores = np.array([10, 0, 0, 1, 5])
    predicted_scores = np.array([.1, .2, .3, 4, 70])
    l = LambdaRankLoss()
    lambdas = l.forward(true_scores,predicted_scores)
    for v in lambdas:
        print(v)
    
    # 5.02794510950437e-31
    # -2.30401702114182e-31
    # -7.974390680400253e-33
    # -8.215937417785356e-33
    # -2.562024807380694e-31
        
    # true_scores = [10, 0, 0, 1, 5]
    # get_good_pair(true_scores)