#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
https://www.jianshu.com/p/2920c97e9e16
"""
import numpy as np
import sklearn
import xgboost as xgb

X_START_IDX = 4
Y_IDX = 3

OUTPUT_FIELDS = "stockno,date,high,high_label,"
#rank_low_model_date_pairwise,rank_low_model,regression_low_model,regression_rank_pairewise_high_label_model,regression_rank_pairewise_low_label_model,multiclass_classification_high_label_model,multiclass_classification_low_label_model,rank_low_model_date_ndcg
MODEL_NAMES = "regression_high_model,multiclass_high_model,multiclass_high_model_mlogloss,train_rank_random,train_rank_date"

def load_fmap():
    with open("schema/fmap.txt",'r') as f :
       return [line.strip() for line in f]

def output_features_weight(bst, model_name):
    "https://zhuanlan.zhihu.com/p/355884348"
    fmap = load_fmap()
    str_types = 'weight,gain,total_gain,cover,total_cover'
    importance_types = str_types.split(",")
    

    fea_weights = {}
    for importance_type in importance_types:
        d = bst.get_score(importance_type=importance_type)
        for k, v in d.items():
            f = fea_weights.get(k, {})
            f[importance_type] = v
            fea_weights[k] = f

    lines = ["fid,fname,"+str_types+"\n"]
    for k, v in fea_weights.items():
        l = []
        l.append(k)
        l.append(fmap[int(k.replace('f',''))])
        # print()
        for t in importance_types:
            l.append(str(v[t]))
        lines.append(",".join(l)+"\n")
    with open("model/features_weight_%s.csv" % (model_name), 'w') as f:
        f.writelines(lines)


def pred(model_name, x_data):
    model_file = "model/%s.json" % (model_name)
    bst = xgb.Booster()
    bst.load_model(model_file)
    output_features_weight(bst, model_name)

    pred_y = bst.predict(x_data)
    return pred_y


def predict_score(data_type="predict"):
    # print("# === %s ===" % (data_type))
    data_file = "data/%s.txt" % (data_type)
    data = np.loadtxt(data_file, delimiter=',')
    str_date = str(int(data[1][0]))
    # print(str_date)
    data_X = np.ascontiguousarray(data[:, X_START_IDX:])
    xg_X = xgb.DMatrix(data_X)

    labels_high = np.ascontiguousarray(data[:, Y_IDX]).reshape(1, -1)
    labels_low = np.ascontiguousarray(data[:, 5]).reshape(1, -1)

    pred_li = []
    model_names = MODEL_NAMES.split(",")
    for model_name in model_names:
        pred_y = pred(model_name, xg_X)
        pred_li.append(pred_y)

        if data_type != "predict":
            labels = labels_high #if "high" in model_name else labels_low
            # for ndcg_k in [5,10,20,30]:
            tmp =  pred_y.reshape(1, -1)
            ndcg5 = sklearn.metrics.ndcg_score(
                labels,tmp, k=5)
            ndcg10 = sklearn.metrics.ndcg_score(
                labels, tmp, k=10)
            ndcg20 = sklearn.metrics.ndcg_score(
                labels, tmp, k=20)
            print("%s %s %s ndcg@5=%0.3f ndcg@10=%0.3f ndcg@20=%0.3f" %
                  (str_date,data_type, model_name, ndcg5,ndcg10,ndcg20))

    lines = [OUTPUT_FIELDS+MODEL_NAMES]
    for i, x in enumerate(data):
        stock_no = str(int(x[1])).zfill(6)
        date = str(int(x[0]))
        fields = [stock_no, date, x[2], x[3]]
        for p in pred_li:
            fields.append(p[i])

        str_fields = ",".join([str(f) for f in fields])
        lines.append(str_fields)

    with open("output/%s_%s.csv" % (data_type,str_date), 'w') as f:
        f.writelines("\n".join(lines))


if __name__ == "__main__":
    # predict_score("all_test")
    predict_score("test")
    predict_score("predict")
