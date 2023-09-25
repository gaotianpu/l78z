#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn

# https://pytorchltr.readthedocs.io/en/stable/getting-started.html

# https://zhuanlan.zhihu.com/p/148262580

# https://www.cnblogs.com/bentuwuying/p/6690836.html

class LambdaRankLoss(nn.Module):
    """
    Get loss from one user's score output
    """
    def forward(
        self, score_predict: torch.Tensor, score_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param score_predict: 1xN tensor with model output score
        :param score_real: 1xN tensor with real score
        :return: Gradient of ranknet
        """
        sigma = 1.0
        score_predict_diff_mat = score_predict - score_predict.t()
        score_real_diff_mat = score_real - score_real.t()
        tij = (1.0 + torch.sign(score_real_diff_mat)) / 2.0
        lambda_ij = torch.sigmoid(sigma * score_predict_diff_mat) - tij
        loss = lambda_ij.sum(dim=1, keepdim=True) - lambda_ij.t().sum(dim=1, keepdim=True)
        return loss



