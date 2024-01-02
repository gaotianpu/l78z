

CREATE TABLE raw_daily(
    trade_date  INT    NOT NULL,
    btc_no    INT    NOT NULL,
    open  DECIMAL(10,4) NOT NULL,  
    high DECIMAL(10,4) NOT NULL,  
    low   DECIMAL(10,4) NOT NULL,
    close     DECIMAL(10,4) NOT NULL,
    amount       DECIMAL(10,4) NOT NULL,
    rate      DECIMAL(10,4) NOT NULL,
    primary key (trade_date,btc_no)
);
CREATE INDEX idx_raw_daily on raw_daily (btc_no, trade_date);
CREATE INDEX idx_raw_daily_rate on raw_daily (btc_no, rate);

/*
sqlite3 data/btc.db <<EOF
.separator ";"
.import data/btc/all_2014_2023.raw.data raw_daily
EOF

sqlite3 data/btc.db <<EOF
.separator ";"
.import data/btc/all_2014_2023.csv raw_with_delta_daily
EOF

*/

/*gen_fields.py 生成*/
CREATE TABLE raw_with_delta_daily(
    trade_date  INT    NOT NULL,
    btc_no    INT    NOT NULL,
    open  DECIMAL(10,4) NOT NULL,  
    high DECIMAL(10,4) NOT NULL,  
    low   DECIMAL(10,4) NOT NULL,
    close     DECIMAL(10,4) NOT NULL,
    amount       DECIMAL(10,4) NOT NULL,
    rate      DECIMAL(10,4) NOT NULL,
    delta_open_open DECIMAL(10,4) NOT NULL,
    delta_open_low DECIMAL(10,4) NOT NULL,
    delta_open_high DECIMAL(10,4) NOT NULL,
    delta_open_close DECIMAL(10,4) NOT NULL,
    delta_low_open DECIMAL(10,4) NOT NULL,
    delta_low_low DECIMAL(10,4) NOT NULL,
    delta_low_high DECIMAL(10,4) NOT NULL,
    delta_low_close DECIMAL(10,4) NOT NULL,
    delta_high_open DECIMAL(10,4) NOT NULL,
    delta_high_low DECIMAL(10,4) NOT NULL,
    delta_high_high DECIMAL(10,4) NOT NULL,
    delta_high_close DECIMAL(10,4) NOT NULL,
    delta_close_open DECIMAL(10,4) NOT NULL,
    delta_close_low DECIMAL(10,4) NOT NULL,
    delta_close_high DECIMAL(10,4) NOT NULL,
    delta_close_close DECIMAL(10,4) NOT NULL,
    delta_amount DECIMAL(10,4) NOT NULL,
    range_price DECIMAL(10,4) NOT NULL,
    rate_1 DECIMAL(10,4) NOT NULL,
    rate_2 DECIMAL(10,4) NOT NULL,
    rate_3 DECIMAL(10,4) NOT NULL,
    rate_4 DECIMAL(10,4) NOT NULL,
    rate_5 DECIMAL(10,4) NOT NULL,
    rate_6 DECIMAL(10,4) NOT NULL,
    primary key (trade_date,btc_no)
);



CREATE TABLE train_data(
    pk_date_btc UNSIGNED BIG INT NOT NULL,
    trade_date  INT    NOT NULL,
    btc_no    CHAR(6)    NOT NULL,
    dataset_type TINYINT NOT NULL DEFAULT 0,  
    highN_rate     DECIMAL(10,4) NOT NULL,
    lowN_rate     DECIMAL(10,4) NOT NULL,
    high1_rate     DECIMAL(10,4) NOT NULL,
    low1_rate     DECIMAL(10,4) NOT NULL,
    data_json   TEXT    NOT NULL,
    primary key (pk_date_btc)
);
CREATE INDEX idx_train_data on train_data (btc_no,trade_date);
CREATE INDEX idx_train_data_1 on train_data (trade_date,btc_no);
CREATE INDEX idx_train_data_2 on train_data (dataset_type, highN_rate);
CREATE INDEX idx_train_data_3 on train_data (dataset_type, lowN_rate);

/*
# 导入数据
sqlite3 data/btc_train.db <<EOF
.separator ";"
.import data/btc/all_train_data.data train_data
EOF

python train_point.py training low1 
python train_point.py training high1 
python train_point.py training lowN 
python train_point.py training highN 
*/