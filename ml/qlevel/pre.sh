#!/bin/bash

# wget ftp://cp01-gaotianpu-dev.epc.baidu.com/home/users/gaotianpu/baidu/personal-code/gaotianpu/realtime_ugc_query/data/query.tmp -O data/query.tmp
python query_level_pre.py > data/query.after_level
python query_res_count_pre.py > data/query.after_res_count
