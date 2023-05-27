
ROOT_PATH=~/Documents/stocks
# ROOT_PATH=~/extdata/stocks

if [ ! -d "$ROOT_PATH/schema/" ];then
    mkdir $ROOT_PATH/
    mkdir $ROOT_PATH/data/
    mkdir $ROOT_PATH/data/history
    mkdir $ROOT_PATH/model/
    mkdir $ROOT_PATH/schema/
    mkdir $ROOT_PATH/log/
    mkdir $ROOT_PATH/output/
fi

cp schema/*   $ROOT_PATH/schema/
cp download_history.py $ROOT_PATH/download_history.py
cp preprocess.py $ROOT_PATH/preprocess.py
cp preprocess_v3.py $ROOT_PATH/preprocess_v3.py
cp preprocess_v4.py $ROOT_PATH/preprocess_v4.py
cp xgboost_data.py $ROOT_PATH/xgboost_data.py
cp train.py $ROOT_PATH/train.py
cp predict.py $ROOT_PATH/predict.py
cp run.sh  $ROOT_PATH/run.sh
cp run_predict.sh  $ROOT_PATH/run_predict.sh