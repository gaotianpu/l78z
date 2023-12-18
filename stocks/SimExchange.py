import os
import sys
import pandas as pd
import sqlite3
import random

class SimExchange():
    def __init__(self,exchangeName="strategy1",startDate=20231201,initAmount=100000.00):
        ''' 模拟交易数，用于策略效果回测
        '''
        # 虚拟交易所名称
        self.exchangeName = exchangeName
        # 账户初始金额
        self.initAmount = initAmount
        #当前交易日期
        self.currentTradeDate = 20231201 #训练数据截止日的下一个交易日？
        
        # 行情数据库
        self.priceDB = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)
        self.startDate = startDate
        self.stocks = None #可交易的股票代码
        self.tradeDates = None #可交易日
        
        # 交易数据的存储库
        self.exchangeDBFile = f"newdb/SimExchange_{exchangeName}.db"
        self.exchangeDB = None #sqlite3.connect("file:"+self.exchangeDBFile, uri=True)
        
    
    def init_SimExchange(self):
        """
        模拟交易所的初始化
        1. 创建sqlite3 db
        2. 读取股票列表、交易日期
        """
        print(self.exchangeDBFile)
        if os.path.exists(self.exchangeDBFile):
            # print(self.exchangeDBFile,"已存在，无需再创建")
            self.exchangeDB = sqlite3.connect("file:"+self.exchangeDBFile, uri=True)
        else:
            # real_high, ?
            # op_type: 操作类型，1买入，-1卖出
            # stock_hands: 和op_type符号保持一致
            # amount: -price*stock_hands*100,和op_type符号相反
            # order_status: 0-失败，1-成功，2-取消
            dbSchema = """CREATE TABLE SimExchange_records(
                order_no  VARCHAR(20)  NOT NULL,
                trade_date  INT    NOT NULL,
                stock_no    CHAR(6)    NOT NULL,
                real_low     DECIMAL(10,2) NOT NULL,
                real_high     DECIMAL(10,2) NOT NULL,
                op_type TINYINT NOT NULL DEFAULT 0,  
                price     DECIMAL(10,2) NOT NULL,
                stock_hands  INT    NOT NULL,
                amount     DECIMAL(10,2) NOT NULL,
                order_status TINYINT NOT NULL DEFAULT 0,
                primary key (order_no)
            );"""
            # idx: stock_no,order_status
            
            self.exchangeDB = sqlite3.connect("file:"+self.exchangeDBFile, uri=True)
            cur = self.exchangeDB.cursor()
            cursor = cur.execute(dbSchema) 
            self.exchangeDB.commit()
            cursor.close()
        
        # 初始化交易日列表
        sql = f"select distinct trade_date from stock_raw_daily where trade_date>={self.startDate}"
        df = pd.read_sql(sql, self.priceDB)
        self.tradeDates = tuple(df['trade_date'].sort_values(ascending=True).tolist()) 
        
        # 初始化股票代码列表
        sql = f"select distinct stock_no from stock_raw_daily  where trade_date>={self.startDate}"
        df = pd.read_sql(sql, self.priceDB)
        self.stocks = tuple(df['stock_no'].tolist())
    
    def set_tradeDate(self,tradeDate):
        self.currentTradeDate = tradeDate
    
    def get_tradeDates(self):
        '''可用于交易的日期，正序排列输出'''
        return self.tradeDates
    
    def get_stocks(self):
        '''可用于交易的股票代码，正序排列输出'''
        return self.stocks
    
    def get_real_prices(self,tradeDate,stocks):
        # df['last_close_price'] = df['CLOSE_price'] - df['change_amount'] 
        strStocks = ",".join([f"'{s}'" for s in stocks])
        sql = f"select stock_no,LOW_price,HIGH_price, (CLOSE_price-change_amount) as Last_close from stock_raw_daily where trade_date={tradeDate} and stock_no in ({strStocks})"
        # print(sql)
        df = pd.read_sql(sql, self.priceDB)
        ret = {}
        for idx,row in df.iterrows():
            ret[row['stock_no']]=(row['LOW_price'],row['HIGH_price'],row['Last_close'])
        return ret 
        
    
    def get_remainingAmount(self,tradeDate):
        """获取交易所状态：可用余额；持仓订单(股票、数量，金额);当日挂单未成交部分(占据了可用金额) 
        可用余额  = 初始金额 - 持仓金额 - 当天挂单的金额
        """
        sql = f"select sum(amount) as amount from SimExchange_records where order_status=1"
        df = pd.read_sql(sql, self.exchangeDB)
        amount = df['amount'][0] #买入股票amount为符号，卖出为正号
        
        #交易当日，买入挂单，但失败的，也占用可用金额
        sql = f"select sum(amount) as amount from SimExchange_records where trade_date={tradeDate} and op_type=1 and order_status=0"
        df = pd.read_sql(sql, self.exchangeDB)
        holdAmount = df['amount'][0]
        
        return self.initAmount + amount + holdAmount
    
    def get_remainingStockHands(self,stocks):
        """股票剩余数量，金额"""
        ret = {}
        strStocks = ",".join([f"'{s}'" for s in stocks])
        sql = f"select stock_no,sum(stock_hands) as stock_hands,sum(amount) as amount from SimExchange_records where stock_no in ({strStocks}) and order_status=1 group by stock_no"
        df = pd.read_sql(sql, self.exchangeDB)
        for idx,row in df.iterrows():
            ret[row['stock_no']]=[row['stock_hands'],row['amount']]
        return ret
    
    def __insertOrders(self,orders):
        '''向数据库中插入订单数据，私有函数，调用前必须在order()函数中判断各种业务逻辑     
        orders的数据结构 [[order_no,trade_date,stock_no,real_low,real_high,op_type,price,stock_hands,amount,order_status]]
        '''
        # 1.检查orders数据是否非法
        if not orders:
            return False,"orders is None"
        for idx,order in enumerate(orders):
            if len(order)!=10 :
                return False, f"{idx} fields count !=9"
            # 0. order_no 不能重复？
            order_no = order[0]
            # 1. trade_date为真实的交易日期
            # 2. stock_no为真实的股票代码
            # 3.4. real_high,real_low
            if order[3]==order[4]:
                return False, f"order_no={order_no},idx={idx} real_high=real_low, up/down stop, cannot buy or sell!"
            # 5. op_type：1买入，-1卖出
            if order[5] not in [-1,1]:
                return False,f"order_no={order_no},idx={idx} op_type not in [-1,1]"
            # 6. price,
            # 7. stock_hands,
            # 8. amount,
            # 9. order_status,订单状态，1成交，0未成交
            if order[9] not in [0,1]:
                return False, f"order_no={order_no},idx={idx} order_status not in [0,1]"
        
        # 2. 执行订单插入的操作
        cur = self.exchangeDB.cursor()
        try:
            cur.executemany("INSERT INTO SimExchange_records(?,?,?,?,?,?,?,?,?)", orders)  # commit_id_list上面已经说明
            self.exchangeDB.commit()
        except:
            print("exception")
            self.exchangeDB.rollback()
            cur.close()
            
        return True,"sucess"
    
    def order(self,tradeDate,orders):
        '''挂单操作，操作类型：买入，卖出；订单状态：成功、失败
        in: stock_no,op_type,price,stock_hands
        out: order_no,trade_date,stock_no,real_low,real_high,op_type,price,stock_hands,amount,order_status
        '''
        stocks = [o[0] for o in orders]
        # 账号剩余金额
        remainingAmount = self.get_remainingAmount(tradeDate)
        # 账户持股多少手 
        stockHands = self.get_remainingStockHands(stocks) 
        # 市场价格
        realPrices = self.get_real_prices(tradeDate,stocks)
        
        ret = []
        for order in orders:
            (stockNo,opType,price,stockHands) = tuple(order)
            
            r = random.randint(1,9999)
            orderNo = f"{tradeDate}{stockNo}{r}"
            rprice = realPrices.get(stockNo)
            if not rprice:
                print("{tradeDate} {stockNo} cannot get real prices")
                continue 
            
            # opType, 1买入，-1卖出， stockHands,1手=100股
            stockHands = opType*stockHands
            # 买入的金额为负(付钱出去)，卖出的金额为正(把股票变回钱)。stockHands和amount符号相反
            amount = -stockHands*price*100 
            order_status = 0
            if rprice[0]==rprice[1]: #涨跌停
                order_status = 0
            if opType>0 and price>rprice[0]: #买入操作时,出价比当日最低价要高
                #验证账户余额是否大于amount
                order_status = 1 
            if opType<0 and price<rprice[1]: #卖出操作时,出价比当日最高价要低
                #验证账号余票是否大于stockHands
                order_status = 1 
            
            ret.append([orderNo,tradeDate,stockNo,rprice[0],rprice[1],opType,price,stockHands,amount,order_status])
        self.__insertOrders(ret)
            

def test():
    exchange = SimExchange("strategy1")
    exchange.init_SimExchange()
    
    # tradeDates = exchange.get_tradeDates()
    # stocks = exchange.get_stocks()
    # print(len(tradeDates),tradeDates[:10])
    # print(len(stocks),stocks[:10])
    
    tradeDate = 20231214
    stocks = ['000001', '000002', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011']
    ret = exchange.get_real_prices(tradeDate,stocks)
    print(ret)

if __name__ == "__main__":
    test()