# 加密货币

## 一、历史数据
下载、处理
https://www.cnblogs.com/sljsz/p/16406375.html

大
https://www.cnblogs.com/sljsz/p/13866396.html

天级、小时级、分钟级？
http://coinmarketcap.com

## 二、API接口
https://github.com/ccxt/ccxt
https://github.com/ccxt/ccxt/wiki/

https://zhuanlan.zhihu.com/p/348655522

https://www.cnblogs.com/sljsz/p/17566779.html

公共 API, 具有获取所有交易所公开信息的权限，无需注册用户账户，也无需 API 密钥
私有API,

https://www.fmz.com/square， 感觉是ccxt基础上搞得


## 三、策略？
1. 趋势操作，用之前股票选股模型的思路搞
2. 如何做空？
3. 美元指数等的相关关系


## 四、回测 backTrader
### 1. 回测框架选型
https://zhuanlan.zhihu.com/p/406708630
* backtrader
* PyAlgoTrade，数字货币？
* zipline(线上？)、
* vnpy
* 量化平台有Quantopian(国外)、聚宽、万矿、优矿、米筐、掘金等

### 2. backtrader:
https://github.com/mementum/backtrader
https://www.backtrader.com/docu/
https://github.com/Skinok/backtrader-pyqt-ui

中文介绍：
https://zhuanlan.zhihu.com/p/261735195
https://zhuanlan.zhihu.com/p/122183963 一
https://zhuanlan.zhihu.com/p/140425363 二
https://zhuanlan.zhihu.com/p/133623610 三
https://zhuanlan.zhihu.com/c_1189276087837011968


backtrader的核心组件：
1. 数据加载(Data Feed)：将交易策略的数据加载到回测框架中。
2. 交易策略(Strategy)：该模块是编程过程中最复杂的部分，需要设计交易决策，得出买入/卖出信号。
3. 回测框架设置(Cerebro)：需要设置(i)初始资金(ii)佣金(iii)数据馈送(iv)交易策略(v)交易头寸大小。
4. 运行回测：运行Cerebro回测并打印出所有已执行的交易。
5. 评估性能(Analyzers):以图形和风险收益等指标对交易策略的回测结果进行评价。

缺点：
1. 可能存在某些内存管理上的bug,导致大量股票回测的时候，出现堵塞和崩溃。?



其他:
https://www.zhihu.com/question/265096151

https://zhuanlan.zhihu.com/p/53301040


vpn:
1. 历史数据下载
2. 各个交易所注册
3. api接口调试