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

/* DROP TABLE stock_raw_daily; 
update stock_raw_daily set change_rate=replace(change_rate,'%',''),TURNOVER_rate=replace(TURNOVER_rate,'%','') where stock_no='002913' and trade_date='20171201'
TURNOVER_rate = '-' 的情况？
select trade_date,stock_no,OPEN_price,CLOSE_price,change_amount,change_rate,LOW_price,HIGH_price,TURNOVER,TURNOVER_amount,TURNOVER_rate from stock_raw_daily_2 order by trade_date,stock_no;
*/
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
    primary key (trade_date,stock_no)
);
CREATE INDEX idx_stock_raw_daily on stock_raw_daily (stock_no, trade_date);

/*stock_with_delta_daily
change_amount   DECIMAL(10,2) NOT NULL,
change_rate     DECIMAL(10,2) NOT NULL,
*/
CREATE TABLE stock_with_delta_daily(
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL,
    OPEN_price  DECIMAL(10,2) NOT NULL,  
    CLOSE_price DECIMAL(10,2) NOT NULL,
    LOW_price       DECIMAL(10,2) NOT NULL,
    HIGH_price      DECIMAL(10,2) NOT NULL,
    TURNOVER    UNSIGNED BIG INT NOT NULL,   /*成交量*/ 
    TURNOVER_amount  FLOAT NOT NULL, /*成交金额*/
    TURNOVER_rate   DECIMAL(10,2) NOT NULL, /*换手率*/
    delta_OPEN_OPEN      DECIMAL(10,2) NOT NULL,
    delta_OPEN_CLOSE      DECIMAL(10,2) NOT NULL,
    delta_OPEN_LOW      DECIMAL(10,2) NOT NULL,
    delta_OPEN_HIGH      DECIMAL(10,2) NOT NULL,
    delta_CLOSE_OPEN      DECIMAL(10,2) NOT NULL,
    delta_CLOSE_CLOSE      DECIMAL(10,2) NOT NULL,
    delta_CLOSE_LOW      DECIMAL(10,2) NOT NULL,
    delta_CLOSE_HIGH      DECIMAL(10,2) NOT NULL,
    delta_LOW_OPEN      DECIMAL(10,2) NOT NULL,
    delta_LOW_CLOSE      DECIMAL(10,2) NOT NULL,
    delta_LOW_LOW      DECIMAL(10,2) NOT NULL,
    delta_LOW_HIGH      DECIMAL(10,2) NOT NULL,
    delta_HIGH_OPEN      DECIMAL(10,2) NOT NULL,
    delta_HIGH_CLOSE      DECIMAL(10,2) NOT NULL,
    delta_HIGH_LOW      DECIMAL(10,2) NOT NULL,
    delta_HIGH_HIGH      DECIMAL(10,2) NOT NULL,
    delta_TURNOVER      DECIMAL(10,2) NOT NULL,
    delta_TURNOVER_amount      DECIMAL(10,2) NOT NULL,
    delta_TURNOVER_rate      DECIMAL(10,2) NOT NULL,
    range_base_lastclose      DECIMAL(10,2) NOT NULL,
    range_base_open      DECIMAL(10,2) NOT NULL,
    zscore_TURNOVER      DECIMAL(10,2) NOT NULL,
    zscore_amount      DECIMAL(10,2) NOT NULL,
    primary key (trade_date,stock_no)
);
CREATE INDEX idx_stock_with_delta_daily on stock_with_delta_daily (stock_no, trade_date);




/*
stock_date_minmax.txt
select trade_date,max(TURNOVER) as max_TURNOVER,min(TURNOVER) as min_TURNOVER,max(TURNOVER_amount) as max_amount,min(TURNOVER_amount) as min_amount from stock_raw_daily group by trade_date;
*/
CREATE TABLE stock_date_statics(
    trade_date  INT    NOT NULL,
    TURNOVER_mean DECIMAL(10,2) NOT NULL,
    TURNOVER_std DECIMAL(10,2) NOT NULL,
    TURNOVER_min DECIMAL(10,2) NOT NULL,
    TURNOVER_max DECIMAL(10,2) NOT NULL,
    TURNOVER_amount_mean DECIMAL(10,2) NOT NULL,
    TURNOVER_amount_std DECIMAL(10,2) NOT NULL,
    TURNOVER_amount_min DECIMAL(10,2) NOT NULL,
    TURNOVER_amount_max DECIMAL(10,2) NOT NULL,
    primary key (trade_date)
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
    highN_rate     DECIMAL(10,2) NOT NULL,
    high1_rate     DECIMAL(10,2) NOT NULL,
    low1_rate     DECIMAL(10,2) NOT NULL,
    list_label TINYINT NOT NULL DEFAULT 0,    
    data_json   TEXT    NOT NULL,
    primary key (pk_date_stock)
);
CREATE INDEX idx_stock_for_transfomer on stock_for_transfomer (stock_no,trade_date)
CREATE INDEX idx_stock_for_transfomer_1 on stock_for_transfomer (trade_date,stock_no)
CREATE INDEX idx_stock_for_transfomer_2 on stock_for_transfomer (dataset_type, highN_rate);

/* trade_date,stock_no,op_type,price,amount,order_status */
CREATE TABLE SimExchange_records(
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL,
    op_type TINYINT NOT NULL DEFAULT 0,  
    price     DECIMAL(10,2) NOT NULL,
    amount     DECIMAL(10,2) NOT NULL,
    order_status TINYINT NOT NULL DEFAULT 0
);

