/*sqlite3 schema*/
/* stock*/
CREATE TABLE stock_basic_info(
   stock_no        CHAR(6) PRIMARY KEY  NOT NULL,
   start_date       CHAR(8)    NOT NULL,
   stock_name      CHAR(4)      NOT NULL,
   is_drop        INT2  DEFAULT 0
);

/*
update stock_basic_info set is_drop=1 where stock_no in ('000502','000613')
*/

/* DROP TABLE stock_raw_daily; */
CREATE TABLE stock_raw_daily(
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL,   
    TCLOSE      DECIMAL(10,2) NOT NULL,
    HIGH        DECIMAL(10,2) NOT NULL,
    LOW         DECIMAL(10,2) NOT NULL,
    TOPEN       DECIMAL(10,2) NOT NULL,
    LCLOSE      DECIMAL(10,2) NOT NULL,
    CHG         DECIMAL(10,2) NOT NULL,
    PCHG        DECIMAL(10,4) NOT NULL,
    TURNOVER    UNSIGNED BIGINT NOT NULL, /*换手率*/
    VOTURNOVER  FLOAT NOT NULL,   /*成交量*/ 
    VATURNOVER  FLOAT NOT NULL, /*成交金额*/
    TCAP    FLOAT NOT NULL,
    MCAP    FLOAT NOT NULL,
    primary key (trade_date,stock_no)
);
CREATE INDEX idx_stock_raw_daily_2 on stock_raw_daily (stock_no, trade_date);

/* DROP TABLE stock_raw_daily_2; */
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

/*
update stock_raw_daily_2 set change_rate=replace(change_rate,'%',''),TURNOVER_rate=replace(TURNOVER_rate,'%','') where stock_no='002913' and trade_date='20171201'
*/

/*stock的统计信息，价格，成交量等均值和标准差，用于对数据的标准化处理*/
CREATE TABLE stock_statistics_info(
   stock_no        CHAR(6) PRIMARY KEY     NOT NULL,
   update_date    INT    NOT NULL,
   data_json   TEXT    NOT NULL
);

/* DROP TABLE stock_trade_date; */
CREATE TABLE stock_trade_date(
    trade_date  INT PRIMARY KEY NOT NULL,
    stock_count INT NOT NULL DEFAULT 0
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
    data_json   TEXT    NOT NULL,
    primary key (pk_date_stock)
);
CREATE INDEX idx_stock_for_transfomer_2 on stock_for_transfomer (stock_no, trade_date);
CREATE INDEX idx_stock_for_transfomer_3 on stock_for_transfomer (trade_date, stock_no);
CREATE INDEX idx_stock_for_transfomer_4 on stock_for_transfomer (dataset_type);




 .separator ","
 .import schema/stocks.txt stock_basic_info
 .import data/history_all.csv stock_raw_daily
--  .import data/history/000725.csv stock_raw_daily

