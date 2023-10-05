class LambdaRankLoss2(nn.Module):
    def dcg_f(self,scores):
        v = 0
        for i in range(len(scores)):
            v += (np.power(2, scores[i]) - 1) / np.log2(i+2)  # i+2 is because i starts from 0
        return v
    
    def single_dcg(self,scores, i, j):
        return (np.power(2, scores[i]) - 1) / np.log2(j + 2)
        
        # scores = torch.tensor(scores) 
        # return (torch.pow(2, scores[i]) - 1) / torch.log2(j + 2)

    def ideal_dcg(self,scores):
        # print("sorted:",sorted(scores,reverse=True))
        scores = [score for score in sorted(scores)[::-1]]
        # print("ideal_dcg:",scores)
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
        idcg = self.ideal_dcg(true_scores)
        # print(type(true_scores),type(predicted_scores))
        
        good_ij_pairs = self.get_good_pair(true_scores) 
        
        num_docs = len(true_scores)
        sorted_indexes = np.argsort(predicted_scores)[::-1]
        rev_indexes = np.argsort(sorted_indexes)
        true_scores = true_scores[sorted_indexes]
        predicted_scores = predicted_scores[sorted_indexes]
        
        single_dcgs = {}
        for i,j in good_ij_pairs:
            if (i,i) not in single_dcgs:
                single_dcgs[(i,i)] = self.single_dcg(true_scores, i, i)
            single_dcgs[(i,j)] = self.single_dcg(true_scores, i, j)
            if (j,j) not in single_dcgs:
                single_dcgs[(j,j)] = self.single_dcg(true_scores, j, j)
            single_dcgs[(j,i)] = self.single_dcg(true_scores, j, i)
        
        lambdas = np.zeros(num_docs) 
        for i,j in good_ij_pairs:
            z_ndcg = abs(single_dcgs[(i,j)] - single_dcgs[(i,i)] + single_dcgs[(j,i)] - single_dcgs[(j,j)]) / idcg
            rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
            # rho_complement = 1.0 - rho
            lambda_val = z_ndcg * rho
            lambdas[i] += lambda_val
            lambdas[j] -= lambda_val 
        
        return lambdas

# loss_lambdas = loss_fn(predict_scores.detach().numpy(), labels.numpy())   
        
        # # Back propagation 
        # # loss.backward() 
        # learning_rate = 0.0001
        # model.zero_grad()
        # lambdas_torch = torch.Tensor(loss_lambdas) #.view((len(lambdas), 1))
        # predict_scores.backward(lambdas_torch, retain_graph=True)  # This is very important. Please understand why?
        # # with torch.no_grad():
        # for param in model.parameters():
        #     # print(param,param.grad)
        #     if param.grad:
        #         print(param.grad.data)
        #         param.data.add_(param.grad.data * learning_rate)  
        
        # optimizer.step()
        # optimizer.zero_grad() 
        
        # # 计算ndcg
        # y_true = np.expand_dims(labels.numpy(),axis=0)
        # y_predict = np.expand_dims(predict_scores.detach().numpy(),axis=0)
        
        # ndcg = ndcg + ndcg_score(y_true,y_predict)
        # ndcg_5 = ndcg_5 + ndcg_score(y_true,y_predict,k=5)
        # ndcg_3 = ndcg_3 + ndcg_score(y_true,y_predict,k=3)
        
        # if batch % 30 == 0:
        #     avg_ndcg = ndcg / (batch + 1) 
        #     avg_ndcg_5 = ndcg_5 / (batch + 1) 
        #     avg_ndcg_3 = ndcg_3 / (batch + 1) 
        #     current = batch + 1 #* len(predict_scores)
        #     # loss, current = loss.item(), (batch + 1) * len(predict_scores)
        #     print(f" avg_ndcg: {avg_ndcg:>7f}  , avg_ndcg_5: {avg_ndcg_5:>7f}  , avg_ndcg_3: {avg_ndcg_3:>7f}   [{epoch:>5d}  {current:>5d}/{size:>5d}]") 
# if op_type == "loss":
    #     true_scores = np.array([10, 0, 0, 1, 5])
    #     predicted_scores = np.array([.1, .2, .3, 4, 70])
    #     # true_scores = torch.tensor([10, 0, 0, 1, 5])
    #     # predicted_scores = torch.tensor([.1, .2, .3, 4, 70])
    #     loss_fn = LambdaRankLoss()
    #     lambdas = loss_fn(true_scores,predicted_scores)   
    #     # lambdas = l.forward(true_scores,predicted_scores)
    #     for v in lambdas:
    #         print(v)
    # if op_type == "tmp":  
    #     # dataset = StockPairDataset('train')
    #     # a,b = next(iter(dataset))
    #     # print(a.shape,b.shape) # torch.Size([20, 9]) torch.Size([20, 9])
    #     # dataloader = DataLoader(dataset, batch_size=3) 
    #     # a,b = next(iter(dataloader))
    #     # print(a.shape,b.shape) # torch.Size([3, 20, 9]) torch.Size([3, 20, 9])
        
    #     dataset = StockListDataset('train')
    #     labels,data = next(iter(dataset))
    #     print(labels.shape,data.shape) # torch.Size([24]) torch.Size([24, 20, 9])
        
    #     dataloader = DataLoader(dataset, batch_size=1)
    #     labels,data = next(iter(dataloader))
    #     print(labels.shape,data.shape) #
    #     labels = torch.squeeze(labels)
    #     data = torch.squeeze(data)
    #     print(labels.shape,data.shape) #
    #     print(labels.dtype,data.dtype)