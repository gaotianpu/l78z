/*sqlite3 schema*/
/* stock*/
CREATE TABLE stock_basic_info(
   stock_no        CHAR(6) PRIMARY KEY     NOT NULL,
   start_date       CHAR(8)    NOT NULL,
   stock_name      CHAR(4)      NOT NULL,
   is_drop        INT2  DEFAULT 0
);

/* DROP TABLE stock_trade_date; */
CREATE TABLE stock_trade_date(
    trade_date  INT PRIMARY KEY NOT NULL,
    stock_count INT NOT NULL DEFAULT 0
);

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
    TURNOVER    UNSIGNED BIGINT NOT NULL, /*成交量*/
    VOTURNOVER  FLOAT NOT NULL,  /*成交金额*/
    VATURNOVER  FLOAT NOT NULL, /*？*/
    TCAP    FLOAT NOT NULL,
    MCAP    FLOAT NOT NULL,
    primary key (trade_date,stock_no)
);
CREATE INDEX idx_stock_raw_daily_2 on stock_raw_daily (stock_no, trade_date);

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


CREATE TABLE stock_for_transfomer_test(
    pk_date_stock UNSIGNED BIG INT NOT NULL,
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL,
    dataset_type TINYINT NOT NULL DEFAULT 0,   
    data_json   TEXT    NOT NULL,
    primary key (pk_date_stock)
);
CREATE INDEX idx_stock_for_transfomer_test_2 on stock_for_transfomer_test (stock_no, trade_date);
CREATE INDEX idx_stock_for_transfomer_test_3 on stock_for_transfomer_test (trade_date, stock_no);


/* DROP TABLE stock_for_xgb ;*/

CREATE TABLE stock_for_xgb(
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL, 
    y_high      DECIMAL(10,3) NOT NULL,  
    y_low      DECIMAL(10,3) NOT NULL,  
    label_high  int NOT NULL,  
    label_low   int NOT NULL,  
    f_0      DECIMAL(10,3) NOT NULL,
    f_1      DECIMAL(10,3) NOT NULL,
    f_2      DECIMAL(10,3) NOT NULL,
    f_3      DECIMAL(10,3) NOT NULL,
    f_4      DECIMAL(10,3) NOT NULL,
    f_5      DECIMAL(10,3) NOT NULL,
    f_6      DECIMAL(10,3) NOT NULL,
    f_7      DECIMAL(10,3) NOT NULL,
    f_8      DECIMAL(10,3) NOT NULL,
    f_9      DECIMAL(10,3) NOT NULL,
    f_10      DECIMAL(10,3) NOT NULL,
    f_11      DECIMAL(10,3) NOT NULL,
    f_12      DECIMAL(10,3) NOT NULL,
    f_13      DECIMAL(10,3) NOT NULL,
    f_14      DECIMAL(10,3) NOT NULL,
    f_15      DECIMAL(10,3) NOT NULL,
    f_16      DECIMAL(10,3) NOT NULL,
    f_17      DECIMAL(10,3) NOT NULL,
    f_18      DECIMAL(10,3) NOT NULL,
    f_19      DECIMAL(10,3) NOT NULL,
    f_20      DECIMAL(10,3) NOT NULL,
    f_21      DECIMAL(10,3) NOT NULL,
    f_22      DECIMAL(10,3) NOT NULL,
    f_23      DECIMAL(10,3) NOT NULL,
    f_24      DECIMAL(10,3) NOT NULL,
    f_25      DECIMAL(10,3) NOT NULL,
    f_26      DECIMAL(10,3) NOT NULL,
    f_27      DECIMAL(10,3) NOT NULL,
    f_28      DECIMAL(10,3) NOT NULL,
    f_29      DECIMAL(10,3) NOT NULL,
    f_30      DECIMAL(10,3) NOT NULL,
    f_31      DECIMAL(10,3) NOT NULL,
    f_32      DECIMAL(10,3) NOT NULL,
    f_33      DECIMAL(10,3) NOT NULL,
    f_34      DECIMAL(10,3) NOT NULL,
    f_35      DECIMAL(10,3) NOT NULL,
    f_36      DECIMAL(10,3) NOT NULL,
    f_37      DECIMAL(10,3) NOT NULL,
    f_38      DECIMAL(10,3) NOT NULL,
    f_39      DECIMAL(10,3) NOT NULL,
    f_40      DECIMAL(10,3) NOT NULL,
    f_41      DECIMAL(10,3) NOT NULL,
    f_42      DECIMAL(10,3) NOT NULL,
    f_43      DECIMAL(10,3) NOT NULL,
    f_44      DECIMAL(10,3) NOT NULL,
    primary key (trade_date,stock_no)
);
CREATE INDEX idx_stock_for_xgb_2 on stock_raw_daily (trade_date,stock_no);

CREATE VIEW stock_for_xgb_view AS
SELECT *,round(16*(y_high+0.6)/(0.818+0.6)) as high_label_1,round(16*(y_low+0.6)/(0.668+0.6)) as high_label_2
FROM  stock_for_xgb;

CREATE TABLE stock_for_rnn(
    trade_date  INT    NOT NULL,
    stock_no    CHAR(6)    NOT NULL, 
    y_high      DECIMAL(10,3) NOT NULL,  
    y_low      DECIMAL(10,3) NOT NULL,  
    label_high  int NOT NULL,  
    label_low   int NOT NULL,  
    f_0      VARCHAR(80) NOT NULL,
    f_1      VARCHAR(80) NOT NULL,
    f_2      VARCHAR(80) NOT NULL,
    f_3      VARCHAR(80) NOT NULL,
    f_4      VARCHAR(80) NOT NULL,
    f_5      VARCHAR(80) NOT NULL,
    f_6      VARCHAR(80) NOT NULL,
    f_7      VARCHAR(80) NOT NULL,
    f_8      VARCHAR(80) NOT NULL,
    f_9      VARCHAR(80) NOT NULL,
    primary key (trade_date,stock_no)
);
CREATE INDEX idx_stock_for_rnn_2 on stock_for_rnn (trade_date,stock_no);

CREATE TABLE stock_for_rnn_pair(
    trade_date  INT    NOT NULL,
    stock_0    CHAR(6)    NOT NULL, 
    stock_1    CHAR(6)    NOT NULL, 
    data_type  int not null,
    label 	int NOT NULL, 
    primary key (trade_date,stock_0,stock_1)
);
CREATE INDEX idx_stock_for_rnn_pair_2 on stock_for_rnn_pair (trade_date);



.separator ";"
.import data/rnn_all_data_new.csv stock_for_rnn

/*test*/
.import rnn_all_data_new_1w.csv stock_for_transfomer

.import data/rnn_all_data_new.csv stock_for_transfomer 
/* 13:22 ~ 14:15 */


 .separator ","
 .import schema/stocks.txt stock_basic_info
 .import data/history_all.csv stock_raw_daily
--  .import data/history/000725.csv stock_raw_daily

