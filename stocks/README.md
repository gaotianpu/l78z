# 股票投资

## 一. 预测什么
* 周期
  * 大趋势的判断：7~8年(上证指数的峰值间距？)是一个中国股市牛熊转换的完整周期？ 
  未来2 ~ 3年处在什么位置？
  * 中期：半年
  * 短期： 一周？
* 板块?
* 个股?
* 选股策略：未来n天的股票涨跌幅？n=5 ?
* 交易策略：w

## 二、思路
寻找未来n天内，涨幅最高的topn ？
1. point-wise， 回归问题
2. pair-wise，pair对二分类问题
3. list-wise ，排序

## 三、特征
1. 统计思路，
    1. 每一只股票全部的最高、最低、中位数、四分数？ 
    1. 一段时间的最高、最低，中位数，四分位数？
    1. 成交量
    1. 成交金额？

ETF？
* [深市ETF](http://www.szse.cn/market/product/list/etfList/index.html)
* [沪市ETF](http://www.sse.com.cn/assortment/fund/etf/list/)
 

1. label如何分档，分几档？
a. 完全平均分，每个层级档位均匀；尝试分10档，20档？  
b. 先平均分4分，top部分再接着分4分?

分6档？
label   分位  high   low
6 75%  0.153396  -0.005885
5 50%  0.108310  -0.016234 
4 25%  0.086307  -0.034516
3 75%  0.07      -0.014298
2 50%  0.03      -0.032202
1 25%  0.01      -0.060422


2. 分组方式
a. 完全随机分组
b. 按日期分组

3. 每组样本构造
如何能找到比较涨幅较高的？

top10,


## 特征
过去20天最高、最低价，距离最高最低家的天数，过去5天收盘价，成交量，成交金额


使用sqlite，按日期、按股票筛选，速度更快些？


## 流程
1. 下载更新数据
2. 训练更新模型、发出评估结论
3. 预测未来一周情况



preprocess.py:105: RuntimeWarning: invalid value encountered in double_scalars
  (df_past.loc[i]['TURNOVER'] - mean_TURNOVER)/mean_TURNOVER)