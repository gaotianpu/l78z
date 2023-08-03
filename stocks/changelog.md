# 开发日志

### 2023.8.3
1. 居家赋闲，重新把股票预测模型再跑起来，发现之前依赖的网易的股票历史数据下载链接已经失效了，需要寻找新的历史数据下载方式
2. 使用Transformer机制做预测模型(2022.7月已经尝试过RNN，输入格式差不多，仍需再梳理)
3. 

### 2022.7.23
1. xgboost group 抽样逻辑需要变变？
2. 

### 2022.7.22 
感觉最近一段时间这几个板块涨的太多了，先停掉定投
516160	新能源	1200
159857	光伏etf	1100
159755	电池etf	2000
159710	智能电车etf	1200

### 2022.7.13 
ETF基金定投，每周一次,每周总计约2.31w.
159920	恒生etf	3000

513050	中概互联	1005
517350	中概科技	3000
159740	恒生科技etf	1600

512480	半导体	1200
159995	芯片etf	1500
515050	5GETF	1500

516160	新能源	1200
159857	光伏etf	1100
159755	电池etf	2000
159710	智能电车etf	1200

159949	创业板50etf	1200
512710	军工龙头etf	1200

512510	etf500	1200
159892	恒生医药etf	1200


### 2022.7.11
1. rnn的数据生成效率，需要优化下

### 2022.7.8
1. 开发聚类、可视化的代码

### 2022.7.7
1. 股票交易数据，下载到的都是没有除权的？导致在计算价格，成交量统计指标时，有些不准确?

### 2022.7.6
1. 开发一些普通的检索工具，例如：
a. 当前价格，成交量基于过去n日内最高最低价的归一化，价格和成交量的方差
b. 基于这4个特征(价格、成交量归一化，方差)的聚类：2**4=16个区间？
c. 基于这4个特征，向量相似度查询？
c. 图示化的展示这些数据？

2. TODO:各种etf编号，历史数据下载？
http://www.sse.com.cn/assortment/fund/etf/list/


### 2022.7.5
1. pair-wise 按照diff>0.08调整,epho=7效果好些
train: 0-15624, Avg loss: 0.690372 , Avg correct:0.525379
train: 1-15624, Avg loss: 0.682803 , Avg correct:0.563640
train: 2-15624, Avg loss: 0.681678 , Avg correct:0.566728
train: 3-15624, Avg loss: 0.681017 , Avg correct:0.568421
train: 4-15624, Avg loss: 0.680097 , Avg correct:0.569881
train: 5-15624, Avg loss: 0.677807 , Avg correct:0.574438
train: 6-15624, Avg loss: 0.675774 , Avg correct:0.577516
train: 7-15624, Avg loss: 0.674135 , Avg correct:0.580723
train: 8-15624, Avg loss: 0.670763 , Avg correct:0.584585
train: 9-15624, Avg loss: 0.664754 , Avg correct:0.592934
---
test, 0, Avg loss: 0.690570 correct:0.53127521276474 
test, 1, Avg loss: 0.692313 correct:0.53175288438797 
test, 2, Avg loss: 0.693465 correct:0.5316853523254395 
test, 3, Avg loss: 0.692997 correct:0.5324912071228027 
test, 4, Avg loss: 0.693546 correct:0.5308816432952881 
test, 5, Avg loss: 0.694468 correct:0.5310119390487671 
test, 6, Avg loss: 0.701941 correct:0.5329490900039673 
test, 7, Avg loss: 0.696306 correct:0.5335378050804138 
test, 8, Avg loss: 0.702370 correct:0.5246063470840454 
test, 9, Avg loss: 0.716944 correct:0.519112765789032 


### 2022.7.4
1. rnn-pair 效果不符合预期，怀疑是采样问题？
train: 0-1599, Avg loss: 0.693392 , Avg correct:0.500303
train: 1-1599, Avg loss: 0.693144 , Avg correct:0.501699
train: 2-1599, Avg loss: 0.693101 , Avg correct:0.501953
train: 3-1599, Avg loss: 0.693142 , Avg correct:0.502832
train: 4-1599, Avg loss: 0.693057 , Avg correct:0.504746
train: 5-1599, Avg loss: 0.693141 , Avg correct:0.502773
train: 6-1599, Avg loss: 0.693122 , Avg correct:0.502363
---
test, 0, Avg loss: 0.693141 correct:0.5026311278343201 
test, 1, Avg loss: 0.693141 correct:0.5080364346504211 
test, 2, Avg loss: 0.693139 correct:0.5036839842796326 
test, 3, Avg loss: 0.693134 correct:0.5139217376708984 
test, 4, Avg loss: 0.693065 correct:0.5126556158065796 
test, 5, Avg loss: 0.693116 correct:0.5188136696815491 
test, 6, Avg loss: 0.693119 correct:0.5180241465568542

2. abs(row_0['high']-row_1['high'])>0.05, 4978046 
abs(row_0['high']-row_1['high'])>0.1, 1679058  #
abs(row_0['high']-row_1['high'])>0.15, 631702 


>>> 0.1 构造样本情况, 情况稍微好些
train: 0-26235, Avg loss: 0.683738 , Avg correct:0.558567
train: 1-26235, Avg loss: 0.672759 , Avg correct:0.586452
train: 2-26235, Avg loss: 0.668881 , Avg correct:0.594195
train: 3-26235, Avg loss: 0.665901 , Avg correct:0.599665
train: 4-26235, Avg loss: 0.655168 , Avg correct:0.613433
train: 5-26235, Avg loss: 0.646954 , Avg correct:0.622402
train: 6-26235, Avg loss: 0.639956 , Avg correct:0.629025
train: 7-26235, Avg loss: 0.632431 , Avg correct:0.637354
train: 8-26235, Avg loss: 0.625402 , Avg correct:0.645646
train: 9-26235, Avg loss: 0.618642 , Avg correct:0.654715
---
test, 0, Avg loss: 0.699589 correct:0.5157352685928345 
test, 1, Avg loss: 0.704332 correct:0.5215209722518921 
test, 2, Avg loss: 0.707384 correct:0.5179579854011536 
test, 3, Avg loss: 0.712304 correct:0.5156617760658264 
test, 4, Avg loss: 0.720266 correct:0.5243425965309143 
test, 5, Avg loss: 0.724666 correct:0.5241373777389526 
test, 6, Avg loss: 0.729496 correct:0.5200536251068115 
test, 7, Avg loss: 0.748100 correct:0.5172353386878967 
test, 8, Avg loss: 0.765581 correct:0.5194138884544373 
test, 9, Avg loss: 0.767704 correct:0.5199914574623108 


### 2022.7.2
1. 基于rnn的pair-wise方式
除原始的采样过后的数据，增加一个文件rnn_group_pairs.txt,用于存储pair对，label(1,0),idx_1,idx2,
2. TODO:需要一个test集合，采用ndcg评估效果如何？test集合可以预测未来3天的？
3. 最低价>昨日收盘价的比例有多少？

### 2022.7.1
1. 开盘价大于昨日的收盘价，补缺后上来，可以买入？ 

tips:
1. 两个pandas的dataframe按某列作为key值做数据merge

### 2022.6.30
1. 合并xgboost 和 rnn 的预测结果？ 基于预测的结果topn，早上开盘10分钟后跑数据，关注开盘涨了的；
2. rnn 训练模型的热加载？
3. 需要有个炒股日记，记录下重大的反思？

tips:
1. 最后的全连接层，由1层扩展到2层，中间加了dropout和relu,效果变好了些，另外，数据量不需要那么大。

windows开发环境
vscode + wsl , 还算可以，无需再使用windows的cmd和powershell，新的目录迁移完后，也可以把windows下安装的anconda卸载了。

### 2022.6.29
1. xgboost 样本权重设定？ 过采样/降采样
    https://www.freesion.com/article/23801019561/
    按日期，20个一组，

### 2022.6.28
1. rnn point-wise
    a. 加载模型，跑要预测的数据？
    b. 训练收敛的图示化表示，用多少数据可以达到收敛的程度？
        1. TensorBoardX (tensor2numpy,刷新频率，先存储到本地再可视化)
            https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html?highlight=tensorboard
            https://pytorch.org/docs/stable/tensorboard.html
            https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
        2. Visdom 
            https://zhuanlan.zhihu.com/p/147158851
            https://blog.csdn.net/dongjinkun/article/details/114973401
    c. 对比rnn,lstm,gru 三种方式？
    d. 数据量可能不需要那么大？

问题：
1. 回归数值集中在中间的一坨，似乎没多大的参考意义,pair-wise,list-wise才是王道？
符合正态分布，pair-wise,list-wise构造组的时候，需要特殊设置？ 
a. 第7组应该再拆细些？
b. 采用上采样/下采样的方式达到每组的样本均衡？
c. xgboost的样本权重？
cat data/rnn_train.data | awk -F ';' '{print $5}' | sort | uniq -c
range	count	percent
0	2	0.00%
1	69	0.00%
2	313	0.01%
3	808	0.03%
4	1715	0.05%
5	11621	0.37%
6	279460	8.93%
7	2441281	78.05%
8	341147	10.91%
9	42062	1.34%
10	7383	0.24%
11	1442	0.05%
12	306	0.01%
13	94	0.00%
14	37	0.00%
15	29	0.00%
16	3	0.00%
sum	3127772	100.00%

2. TODO:有200多个预测值是nan值的？
cat data/rnn_predict.data | grep 'nan,0.0,' | wc -l
600532,600533,600535,600536,600537,600538,600539,600540,600543,600545,600546,600547,600548,600549,600550,600551,600552,600555,600556,600557,600558,600559,600560,600561,600562,600781,600782,600783,600784,600785,600787,600789,600790,600791,600792,600793,600794,600795,600796,600797,600798,600800,600801,600802,600803,600804,600805,600807,600808,600809,600810,600811,600812,600814,600815,600816,600817,600818,600819,600820,600821,600822,600823,600824,600825,600826,600827,600828,600829,600830,600831,600833,600834,600835,600836,600837,600838,600839,600841,600843,600844,600845,600846,600847,600848,600850,600851,600896,600897,600898,600900,600901,600903,600905,600906,600908,600909,600916,600917,600918,600919,600926,600928,600929,600933,600936,600939,600956,600958,600959,000780,000782,002118,002119,002120,002121,002122,002123,002124,002125,002126,002127,002128,002129,002130,002131,002132,002133,002134,002135,002136,002137,002138,002139,002140,002141,002142,002144,002145,002146,002147,002148,002149,002150,002151,002152,002153,002154,002155,002156,002157,002158,002159,002160,002411,002412,002413,002414,002415,002416,002417,002418,002419,002776,002777,002778,002779,002780,002781,002782,002783,002785,002786,002787,002788,002789,002790,002791,002792,002793,002795,002796,002797,002798,002799,002800,002801,002802,002803,002805,002806,002807,002808,002809,002810,002811,002812,002813,002815,002816,002817,002818,002819,300108,300109,300110,300111,300112,300113,300114,300115,300116,300117？

tips:
1. python zfill, 填充0


### 2022.6.27
1. rnn point-wise数据格式生成、模型训练, 和xgboost的对比
    a. test集合尽量和xgboost的保持一致，容易对比出效果
2. rnn pair-wise,list-wise 数据生成，模型训练，和xgboost的对比


### 2022.6.26
1. rnn/lstm/gru point-wise模型开发，基于rnn的数据格式生成；
2. 基于昨日xgboost的模型预测结果，分析选股买入？
a. point,pair,list三种预测结果的top30? 3个统一排序后累加？
    似乎list-wise的更好？
b. 每日及时获取这部分的最新股价数据？
c. 买入和卖出时机提醒-没想好？
    昨日的收盘价、最高价、最低价？
    当前价格是否高过昨日最高，是否低于昨日最低？
    高开，补缺位置买入？
    低开，买入？


### 2022.6.25
1. 每日例行化的处理？ 
    a. 下载数据 - 数据预处理 - 模型预测
    b. 训练数据的预处理 - 模型训练
        已经处理过的，就不需要再额外处理了？
2. 交易当天，定时拿到最新数据，对比过往的预测结果，调整持仓情况？


### 2022.6.24
1. 完成基于分档label形式的point,paire,list-wise预测；效果不是特别好？
2. 历史数据下载，merge导入db库；
3. pandas+sqlite，预处理数据？ 


问题：
1. wsl git clone 问题
https://alessandrococco.com/2021/01/wsl-how-to-resolve-operation-not-permitted-error-on-cloning-a-git-repository
2. pandas + sqlite3:
https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html?highlight=sqlite3


一些产出数据：
1.reg_high_y:
[0]     train-rmse:0.43359      vali-rmse:0.43371
[1]     train-rmse:0.39077      vali-rmse:0.39089
[2]     train-rmse:0.35229      vali-rmse:0.35241
[3]     train-rmse:0.31772      vali-rmse:0.31784
...
[77]    train-rmse:0.04650      vali-rmse:0.04659
[78]    train-rmse:0.04650      vali-rmse:0.04659
[79]    train-rmse:0.04650      vali-rmse:0.04659

reg_high_y:eta=0.1,max—depth=6
[0]     train-rmse:0.43359      vali-rmse:0.43371
[1]     train-rmse:0.39077      vali-rmse:0.39089
...
[78]    train-rmse:0.04652      vali-rmse:0.04663
[79]    train-rmse:0.04652      vali-rmse:0.04663

2. reg_high_label
[0]     train-rmse:5.91423      vali-rmse:5.91633
[1]     train-rmse:5.32828      vali-rmse:5.33022
[2]     train-rmse:4.80373      vali-rmse:4.80339
...
[67]    train-rmse:0.51231      vali-rmse:0.55456
[68]    train-rmse:0.51154      vali-rmse:0.55453
[69]    train-rmse:0.51041      vali-rmse:0.55454
[70]    train-rmse:0.50984      vali-rmse:0.55456

3. pair-wise high_y
[0]     train-auc:0.53060       vali-auc:0.52114
[1]     train-auc:0.53619       vali-auc:0.52494
[2]     train-auc:0.53761       vali-auc:0.52569
...
[76]    train-auc:0.57276       vali-auc:0.53503
[77]    train-auc:0.57281       vali-auc:0.53513
[78]    train-auc:0.57359       vali-auc:0.53523
[79]    train-auc:0.57365       vali-auc:0.53528

4. pair-wise high_label 
max_depth = 10
[0]     train-auc:0.51718       vali-auc:0.51057
[1]     train-auc:0.52294       vali-auc:0.51182
[2]     train-auc:0.52367       vali-auc:0.51250
[3]     train-auc:0.52428       vali-auc:0.51280
...
[76]    train-auc:0.54453       vali-auc:0.51650
[77]    train-auc:0.54451       vali-auc:0.51660
[78]    train-auc:0.54471       vali-auc:0.51667
[79]    train-auc:0.54461       vali-auc:0.51667

max_depth = 20
[0]     train-auc:0.57014       vali-auc:0.50511
[1]     train-auc:0.59353       vali-auc:0.50536
...
[77]    train-auc:0.67136       vali-auc:0.51128
[78]    train-auc:0.67143       vali-auc:0.51138
[79]    train-auc:0.67122       vali-auc:0.51141

max_depth=15, eta=0.3
[0]     train-auc:0.53367       vali-auc:0.50754
[1]     train-auc:0.54585       vali-auc:0.50861
...
[86]    train-auc:0.64359       vali-auc:0.50987
[87]    train-auc:0.64376       vali-auc:0.51001
[88]    train-auc:0.64452       vali-auc:0.50995
[89]    train-auc:0.64464       vali-auc:0.50988

m=15,eta=0.1
[88]    train-auc:0.63039       vali-auc:0.51428
[89]    train-auc:0.63047       vali-auc:0.51433
...
[88]    train-auc:0.52094       vali-auc:0.51554
[89]    train-auc:0.52078       vali-auc:0.51555


5. list-wise high_label
[0]     train-auc:0.50771       vali-auc:0.50273
[1]     train-auc:0.50776       vali-auc:0.50275
[2]     train-auc:0.51034       vali-auc:0.50657
[3]     train-auc:0.50936       vali-auc:0.50653
...
[77]    train-auc:0.51450       vali-auc:0.50875
[78]    train-auc:0.51450       vali-auc:0.50875
[79]    train-auc:0.51450       vali-auc:0.50875

'eta': 0.3, 'max_depth': 6
[0]     train-auc:0.50771       vali-auc:0.50273
[1]     train-auc:0.50776       vali-auc:0.50275
...
[88]    train-auc:0.51493       vali-auc:0.50979
[89]    train-auc:0.51493       vali-auc:0.50979

list-wise,ndcg:
[0]     train-ndcg:0.85791      vali-ndcg:0.83747
[1]     train-ndcg:0.85791      vali-ndcg:0.83745
...
[88]    train-ndcg:0.86736      vali-ndcg:0.84610
[89]    train-ndcg:0.86736      vali-ndcg:0.84610
ndcg@200-
[0]     train-ndcg@200-:0.43912 vali-ndcg@200-:0.83747
[1]     train-ndcg@200-:0.43909 vali-ndcg@200-:0.83745
...
[7]     train-ndcg@200-:0.44774 vali-ndcg@200-:0.84073
[8]     train-ndcg@200-:0.44774 vali-ndcg@200-:0.84073
[9]     train-ndcg@200-:0.44774 vali-ndcg@200-:0.84073

pair-wise,ndcg
[0]     train-ndcg:0.86029      vali-ndcg:0.84045
[1]     train-ndcg:0.86118      vali-ndcg:0.84199
...
[9]     train-ndcg:0.86322      vali-ndcg:0.84238
[10]    train-ndcg:0.86337      vali-ndcg:0.84249
ndcg@200-
[0]     train-ndcg@200-:0.45304 vali-ndcg@200-:0.84045
[1]     train-ndcg@200-:0.45775 vali-ndcg@200-:0.84199
...
[9]     train-ndcg@200-:0.46332 vali-ndcg@200-:0.84238
[10]    train-ndcg@200-:0.46326 vali-ndcg@200-:0.84249

point-wise,ndcg:
[0]     train-ndcg:0.93802      vali-ndcg:0.91221
[1]     train-ndcg:0.93861      vali-ndcg:0.91276
[2]     train-ndcg:0.93876      vali-ndcg:0.91294
...
[23]    train-ndcg:0.94032      vali-ndcg:0.91438
[24]    train-ndcg:0.94036      vali-ndcg:0.91426
[25]    train-ndcg:0.94038      vali-ndcg:0.91425
ndcg@200-
[0]     train-ndcg@200-:0.00873 vali-ndcg@200-:0.10208
...
[6]     train-ndcg@200-:0.01760 vali-ndcg@200-:0.11187
[7]     train-ndcg@200-:0.01760 vali-ndcg@200-:0.11185



### 2022.6.23
引入sqlite管理下载的原始数据和用于xgboost的训练数据

sqlite在数据处理的几个步骤中的应用：

1. 下载和存储历史数据
    
    * 初始化时：首次将股票的历史数据全部下载
    * 每日更新：为减少下载量，后续每日增量更新，在和历史数据merge时，引入sqlite会更高效些。 

2. 加工数据产出基于xgboost的数据 
    
    * 训练数据的y值是一个连续值，需要根据标量量化方式将其划分8个离散的分档，这个操作用sqlite的 update 机制更高效些

todo:

1. 创建tableschema: stock_basic_info, stock_raw_daily 
2. 历史数据的批量导入,包括股票基本信息导入，历史数据导入，预处理中间数据
3. 数据的的读取操作，按stock_no取历史数据，按日期取数据
4. 基于sqlite的数据预处理操作脚本

问题：

1. stock_no+date, sqlite3的联合主键？
2. shell: cat data/history/*.csv > data/history_all.csv , 文件尾部没有换行符？怎么补充上？
https://www.yisu.com/zixun/407865.html
find data/history/ -name '*.csv' | xargs sed 'a\' > data/history_all.csv  
3. sqlite3 drop table的速度很慢？标记删除，还是物理删除呢？
重新创建表后，原来的索引还能关联上吗？
4. 导入时间：14:31 ~ wc 3,980,881
5. 预处理过的文件y值只有最大，最小的连续值，需要增加2个对应的离散值，需要先增加2个列，在sqlite中获取2个连续值的最大最小值，再标量量化操作？
6. 0.818,-0.6,  0.668,-0.6
select max(y_high),min(y_high),max(y_low),min(y_low) from stock_for_xgb ;

select y_high,y_low,round(16*(y_high+0.6)/(0.818+0.6)),round(16*(y_low+0.6)/(0.668+0.6)) from stock_for_xgb where trade_date='20220609' limit 800;

update stock_for_xgb set label_high = round(16*(y_high+0.6)/(0.818+0.6)),label_low=round(16*(y_low+0.6)/(0.668+0.6))
from stock_for_xgb where trade_date='20220609' ;

Parse error: ambiguous column name: y_high
  update stock_for_xgb set label_high = round(16*(y_high+0.6)/(0.818+0.6)),label
                                    error here ---^

cat data/all.sort.txt | python tools/insert_col.py > all.sort.txt &