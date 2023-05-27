import pandas as pd 

data_type = "train"
df = pd.read_csv("data/rnn_%s.txt"%(data_type),
        names="date,stock,high,low,high_label,low_label,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16".split(","),
        header=None, dtype={'stock':str,'high_label':int}, sep=";")

df.describe()