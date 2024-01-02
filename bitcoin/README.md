# 加密货币

## 一、历史数据
下载、处理
https://www.cnblogs.com/sljsz/p/16406375.html

大
https://www.cnblogs.com/sljsz/p/13866396.html
https://www.cryptodatadownload.com/data/

天级、小时级、分钟级？
http://coinmarketcap.com

https://www.binance.com/zh-CN/landing/data
https://www.coincarp.com/zh/currencies/bitcoin/history/


## 二、API接口
https://github.com/ccxt/ccxt
https://github.com/ccxt/ccxt/wiki/  ,  https://docs.ccxt.com
https://github.com/ccxt/ccxt/tree/master/examples/py

https://www.cnblogs.com/sljsz/p/17566779.html
https://www.cnblogs.com/sljsz/p/14671844.html

### 1. 基础概念
* exchanges: 交易所
* markets: 一个由常用交易对或符号索引的市场关联数组。在访问此属性之前，应先加载市场。在调用交易所实例上的 loadMarkets() / load_markets() 方法之前，市场不可用。
* symbols:交易所可用的非关联数组（列表），按字母顺序排列。这些是market属性的键。符号从市场加载和重新加载。这个属性是所有市场键的方便简写。
* currencies：一个关联数组（一个字典），按代码（通常为3个或4个字母）列出可进行兑换的货币。货币从市场加载和重新加载。

* Price Tickers, 价格报价包含最近一段时间内（通常为 24 小时）特定市场/交易品种的统计数据。
* Order Book, 

* fetchTickers
* fetchOrderBook
* fetchOHLCV

Exchanges  →  Markets→Symbols→Currencies
交易所 → 市场 → 交易品种 → 货币

spot / margin / future / swap / option /contract


derivative contracts 衍生品合约
* spot 现货
* margin 保证金
* future – for expiring futures contracts that have a delivery/settlement date
* swap – for perpetual swap futures that don't have a delivery date
* option – for option contracts (https://en.wikipedia.org/wiki/Option_contract)
* contract a future, a perpetual swap, or an option

订单聚合的详细程度或层次通常以数字标记，如L1、L2、L3…
* L1：信息量较少，可以快速获取基本信息，即市场价格。在订单簿中看起来就像一个订单。
* L2：最常见的聚合级别，订单量按价格分组。如果两个订单的价格相同，则它们会显示为一个订单，其数量等于它们的总和。这很可能是您在大多数情况下需要的聚合级别。
* L3：最详细的级别，没有聚合，每个订单与其他订单分开。这种LOD在输出中自然包含重复。因此，如果两个订单的价格相同，它们不会被合并在一起，交易所的匹配引擎将决定它们在堆栈中的优先级。你真的不需要L3的详细信息来成功交易。事实上，你很可能根本不需要它。因此，一些交易所不支持它，总是返回聚合的订单簿。

* 现货,
* 合约, futures

"a currency", "an asset", "a coin", "a token", "stock", "commodity", "crypto", "fiat"
“货币”、“资产”、“硬币”、“代币”、“股票”、“商品”、“加密货币”、“法定货币”

"a market", "a symbol", "a trading pair", "a contract"
“市场”、“符号”、“交易对”、“合约”

base / quote currency, 基础货币，报价货币

taker 购买者 / maker  

'timeInForce': 'GTC',         // 'GTC', 'IOC', 'FOK', 'PO'
* 'GTC' = Good Until Cancel（ed），订单将保留在订单簿上，直到匹配或取消。
* 'IOC' = 立即或取消，订单必须立即匹配并部分或全部执行，未执行的剩余部分被取消（或整个订单被取消）。
* 'FOK' = Fill Or Kill，订单必须立即完全成交并关闭，否则整个订单将被取消。
* 'PO' = 仅发布，订单要么作为制造商订单下达，要么被取消。这意味着订单必须在订单簿上至少处于未成交状态的时间。统一作为一种选择是一项正在进行的工作，统一交易所具有.POtimeInForceexchange.has['createPostOnlyOrder'] == True


https://zhuanlan.zhihu.com/p/348655522

https://www.cnblogs.com/sljsz/p/17566779.html

公共 API, 具有获取所有交易所公开信息的权限，无需注册用户账户，也无需 API 密钥
私有API,

https://www.fmz.com/square， 感觉是ccxt基础上搞得

示例：
https://github.com/yinruiqing/ccxt_notes/blob/master/%E5%A5%97%E5%88%A9%E5%88%86%E6%9E%90.ipynb

如何查看交易所是否支持sandbox？

## 三、策略？
1. 趋势操作，用之前股票选股模型的思路搞
2. 如何做空？
3. 美元指数等的相关关系

策略基线：
https://www.backtrader.com/blog/2019-06-13-buy-and-hold/buy-and-hold/
1. 买入持有到最后一天
2. 定投


## 四、回测 backTrader
### 1. 回测框架选型
https://zhuanlan.zhihu.com/p/406708630
* backtrader
* PyAlgoTrade -> basana ，数字货币？
* zipline(线上？)、
* vnpy
* 量化平台有Quantopian(国外)、聚宽、万矿、优矿、米筐、掘金等

### 2. backtrader
https://github.com/mementum/backtrader
https://www.backtrader.com/docu/
https://github.com/Skinok/backtrader-pyqt-ui

中文介绍：
https://zhuanlan.zhihu.com/p/261735195
https://zhuanlan.zhihu.com/p/122183963 一
https://zhuanlan.zhihu.com/p/140425363 二
https://zhuanlan.zhihu.com/p/133623610 三
https://zhuanlan.zhihu.com/c_1189276087837011968



order的数据结构？
order.executed
* price #成交价格
* value #cost
* comm  #佣金

order.status 订单状态 (每个状态的数值？)
https://www.backtrader.com/docu/order/#order-status-values
* Created
* Submitted 
* Accepted
* Partial
* Completed
* Rejected
* Margin ？
* Canceled
* Expired


exectype: #order execution logic
* Market, 市场价成交，Opening price of the next set
* Close,  Using the close price of the next bar
* Limit, 限价，通常比市价高 
* Stop, 信号产生，且继续上涨时，买入
* StopLimit, 止损限价
* StopTrail
    * trailamount
    * trailpercent
* StopTrailLimit
* OCO, One Cancel Others

def buy(self, data=None,
            size=None, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None,
            parent=None, transmit=True,
            **kwargs):
def sell(self, data=None,
             size=None, price=None, plimit=None,
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None,
             parent=None, transmit=True,
             **kwargs):


bar_executed？

Position：
* Size: 1 #订单数量？
* Price: 43293.13 #当前价格
* Price orig: 0.0 #？
* Closed: 0
* Opened: 1
* Adjbase: 41243.83 #

trade： # OPERATION PROFIT 营业额
* isclosed
* pnl  # GROSS 毛利润
* pnlcomm  # NET 净利润


sizer: ？ 
* stake，赌注，买多少？
* cerebro.addsizer(bt.sizers.FixedSize, stake=10)
* self.sizer.setsizing(self.params.stake)

backtrader的核心组件：
1. 数据加载(Data Feed)：将交易策略的数据加载到回测框架中。
2. 交易策略(Strategy)：该模块是编程过程中最复杂的部分，需要设计交易决策，得出买入/卖出信号。
3. 回测框架设置(Cerebro)：需要设置(i)初始资金(ii)佣金(iii)数据馈送(iv)交易策略(v)交易头寸大小。
4. 运行回测：运行Cerebro回测并打印出所有已执行的交易。
5. 评估性能(Analyzers):以图形和风险收益等指标对交易策略的回测结果进行评价。

缺点：
1. 可能存在某些内存管理上的bug,导致大量股票回测的时候，出现堵塞和崩溃。?


### 3. Basana
https://github.com/gbeced/basana 
https://basana.readthedocs.io/en/latest/quickstart.html



Stop 止损

backtesting 回测
hence is an algotrading platform
algorithmic trading
indicators， 指标信号

其他:
https://www.zhihu.com/question/265096151

https://zhuanlan.zhihu.com/p/53301040

https://github.com/TA-Lib/ta-lib-python

vpn:
1. 历史数据下载
2. 各个交易所注册
3. api接口调试