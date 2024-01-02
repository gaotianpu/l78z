
fields ="trade_date;btc_no;open;high;low;close;amount;rate;delta_open_open;delta_open_low;delta_open_high;delta_open_close;delta_low_open;delta_low_low;delta_low_high;delta_low_close;delta_high_open;delta_high_low;delta_high_high;delta_high_close;delta_close_open;delta_close_low;delta_close_high;delta_close_close;delta_amount;range_price;rate_1;rate_2;rate_3;rate_4;rate_5;rate_6".split(";")
for f in fields:
    print(f,"DECIMAL(10,4) NOT NULL,")