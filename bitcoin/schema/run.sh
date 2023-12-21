

sqlite3 data/btc.db <<EOF
.separator ";"
.import data/btc/all_2014_2023.csv raw_daily
EOF
