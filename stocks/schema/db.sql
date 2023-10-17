/*sqlite3 schema */
/*
stock
update stock_basic_info set is_drop=1 where stock_no in ('000502','000613')
*/
CREATE TABLE stock_basic_info(
   stock_no        CHAR(6) PRIMARY KEY  NOT NULL,
   start_date       CHAR(8)    NOT NULL,
   stock_name      CHAR(4)      NOT NULL,
   is_drop        INT2  DEFAULT 0
); 

/* 存放从网站上下载到的数据，last_close以后的数据是需要加工的
DROP TABLE stock_raw_daily; */
CREATE TABLE stock_raw_daily(
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL,
    OPEN_price  DECIMAL(10,2) NOT NULL,  
    CLOSE_price DECIMAL(10,2) NOT NULL,  
    change_amount   DECIMAL(10,2) NOT NULL,
    change_rate     DECIMAL(10,2) NOT NULL,
    LOW_price       DECIMAL(10,2) NOT NULL,
    HIGH_price      DECIMAL(10,2) NOT NULL,
    TURNOVER    UNSIGNED BIG INT NOT NULL,   /*成交量*/ 
    TURNOVER_amount  FLOAT NOT NULL, /*成交金额*/
    TURNOVER_rate   DECIMAL(10,2) NOT NULL, /*换手率*/
    last_close  DECIMAL(10,2) NOT NULL, /*昨日收盘价*/
    open_rate   DECIMAL(10,2) NOT NULL,
    low_rate    DECIMAL(10,2) NOT NULL,
    high_rate   DECIMAL(10,2) NOT NULL,
    high_low_range  DECIMAL(10,2) NOT NULL,
    open_low_rate   DECIMAL(10,2) NOT NULL,  
    open_high_rate  DECIMAL(10,2) NOT NULL,
    open_close_rate DECIMAL(10,2) NOT NULL,
    primary key (trade_date,stock_no)
);
CREATE INDEX idx_stock_raw_daily_0 on stock_raw_daily (stock_no, trade_date);


CREATE TABLE stock_raw_daily_1(
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL,
    OPEN_price  DECIMAL(10,2) NOT NULL,  
    CLOSE_price DECIMAL(10,2) NOT NULL,  
    change_amount   DECIMAL(10,2) NOT NULL,
    change_rate     DECIMAL(10,2) NOT NULL,
    LOW_price       DECIMAL(10,2) NOT NULL,
    HIGH_price      DECIMAL(10,2) NOT NULL,
    TURNOVER    UNSIGNED BIG INT NOT NULL,   /*成交量*/ 
    TURNOVER_amount  FLOAT NOT NULL, /*成交金额*/
    TURNOVER_rate   DECIMAL(10,2) NOT NULL, /*换手率*/
    last_close  DECIMAL(10,2) NOT NULL, /*昨日收盘价*/
    open_rate   DECIMAL(10,2) NOT NULL,
    low_rate    DECIMAL(10,2) NOT NULL,
    high_rate   DECIMAL(10,2) NOT NULL,
    high_low_range  DECIMAL(10,2) NOT NULL,
    open_low_rate   DECIMAL(10,2) NOT NULL,  
    open_high_rate  DECIMAL(10,2) NOT NULL,
    open_close_rate DECIMAL(10,2) NOT NULL,
    TURNOVER_idx    FLOAT NOT NULL,
    TURNOVER_amount_idx FLOAT NOT NULL,
    TURNOVER_rate_idx   FLOAT NOT NULL,
    change_rate_idx FLOAT NOT NULL,
    last_close_idx  FLOAT NOT NULL,
    open_rate_idx   FLOAT NOT NULL,
    low_rate_idx    FLOAT NOT NULL,
    high_rate_idx   FLOAT NOT NULL,
    high_low_range_idx  FLOAT NOT NULL,
    open_low_rate_idx   FLOAT NOT NULL,
    open_high_rate_idx  FLOAT NOT NULL,
    open_close_rate_idx FLOAT NOT NULL,
    primary key (trade_date,stock_no)
);
CREATE INDEX idx_stock_raw_daily_1_0 on stock_raw_daily_1 (stock_no, trade_date);

/* DROP TABLE stock_raw_daily_2; 
update stock_raw_daily_2 set change_rate=replace(change_rate,'%',''),TURNOVER_rate=replace(TURNOVER_rate,'%','') where stock_no='002913' and trade_date='20171201'
TURNOVER_rate = '-' 的情况？
*/
CREATE TABLE stock_raw_daily_2(
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL,
    OPEN_price  DECIMAL(10,2) NOT NULL,  
    CLOSE_price DECIMAL(10,2) NOT NULL,  
    change_amount   DECIMAL(10,2) NOT NULL,
    change_rate     DECIMAL(10,2) NOT NULL,
    LOW_price       DECIMAL(10,2) NOT NULL,
    HIGH_price      DECIMAL(10,2) NOT NULL,
    TURNOVER    UNSIGNED BIG INT NOT NULL,   /*成交量*/ 
    TURNOVER_amount  FLOAT NOT NULL, /*成交金额*/
    TURNOVER_rate   DECIMAL(10,2) NOT NULL, /*换手率*/
    primary key (trade_date,stock_no)
);
CREATE INDEX idx_stock_raw_daily_2_2 on stock_raw_daily_2 (stock_no, trade_date);


/*stock的统计信息，价格，成交量等均值和标准差，4分位，极值，用于对数据的标准化处理*/
CREATE TABLE stock_statistics_info(
   stock_no        CHAR(6) PRIMARY KEY  NOT NULL,
   update_date    INT    NOT NULL,
   data_json   TEXT    NOT NULL
);

/* DROP TABLE stock_trade_date; 数量，涨跌数，成交量，4分位？ */
CREATE TABLE stock_tradedate_statistics(
    trade_date  INT PRIMARY KEY NOT NULL,
    stock_count INT NOT NULL DEFAULT 0,
    update_date    INT    NOT NULL,
    data_json   TEXT    NOT NULL
);




/* DROP TABLE stock_for_transfomer; dataset_type 默认值=0,验证集=1,测试集=2

select pk_date_stock from stock_for_transfomer where trade_date='20220101' and dataset_type=0;
#random(pk_date_stock,20)
update pk_date_stock set dataset_type=1 where pk_date_stock in ();
*/
CREATE TABLE stock_for_transfomer(
    pk_date_stock UNSIGNED BIG INT NOT NULL,
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL,
    dataset_type TINYINT NOT NULL DEFAULT 0,  
    list_label TINYINT NOT NULL DEFAULT 0,    
    data_json   TEXT    NOT NULL,
    primary key (pk_date_stock)
);
CREATE INDEX idx_stock_for_transfomer_2 on stock_for_transfomer (stock_no, trade_date);
CREATE INDEX idx_stock_for_transfomer_3 on stock_for_transfomer (stock_no,dataset_type,list_label);
CREATE INDEX idx_stock_for_transfomer_4 on stock_for_transfomer (trade_date,dataset_type,list_label);
CREATE INDEX idx_stock_for_transfomer_5 on stock_for_transfomer (dataset_type,list_label);

/* data/stocks_train_4.db */
CREATE TABLE stock_for_boost_v2(
    pk_date_stock UNSIGNED BIG INT NOT NULL,
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL,
    list_label  TINYINT NOT NULL DEFAULT 0,   
    true_score  FLOAT NOT NULL,
    true_high_1 FLOAT NOT NULL,
    true_low_1  FLOAT NOT NULL,
    true_open_rate  FLOAT NOT NULL,
    pair_15     FLOAT NOT NULL,
    list_235    FLOAT NOT NULL,
    point_5     FLOAT NOT NULL,
    point_4     FLOAT NOT NULL,
    pair_11     FLOAT NOT NULL,
    point_high1 FLOAT NOT NULL,
    low1    FLOAT NOT NULL,
    primary key (pk_date_stock)
);
CREATE INDEX stock_for_boost_v2_1 on stock_for_boost_v2 (stock_no,list_label);
CREATE INDEX stock_for_boost_v2_2 on stock_for_boost_v2 (trade_date,list_label);

/*  
.separator ","
.import schema/stocks.txt stock_basic_info
.import data/history_all.csv stock_raw_daily
--  .import data/history/000725.csv stock_raw_daily

# 导出数据时，once会覆盖已有数据，注意！
.once xx.file 
.output xx.file 

*/




