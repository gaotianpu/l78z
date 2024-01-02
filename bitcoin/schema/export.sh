#!/bin/bash



export_func(){
    field=$1
    min_val=$2
    max_val=$3
    dtype=$4

    # echo $field,$dtype,$min_val,$max_val
    echo "select pk_date_btc,${field}_rate from train_data where dataset_type=${dtype} and ${field}_rate>$min_val and ${field}_rate<$max_val order by ${field}_rate desc;"

    sqlite3 data/btc_train.db <<EOF
.separator ";"
.once data/point_${dtype}_$field.csv
select pk_date_btc,${field}_rate from train_data where dataset_type=${dtype} and ${field}_rate>$min_val and ${field}_rate<$max_val order by ${field}_rate desc;
EOF

}

export_func_2(){
    field=$1
    min_val=$2
    max_val=$3

    export_func $field $min_val $max_val 0
    export_func $field $min_val $max_val 1
    export_func $field $min_val $max_val 2

} 


#{'highN_rate': [-13.4735, 19.4128], 
#'lowN_rate': [-17.3754, 14.3518], 
#'high1_rate': [-3.4525, 7.9531], 
#'low1_rate': [-8.2235, 3.7846]}
export_func_2 'highN' -13.4735 19.4128
export_func_2 'lowN' -17.3754 14.3518
export_func_2 'high1' -3.4525 7.9531
export_func_2 'low1' -8.2235 3.7846