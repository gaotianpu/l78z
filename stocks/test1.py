#!/usr/bin/python
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import xgboost as xgb
import matplotlib.pyplot as plt


model_file = "model/rank_high_model_date_ndcg.json"
bst = xgb.Booster()
bst.load_model(model_file)
xgb.plot_tree(bst)
plt.show()