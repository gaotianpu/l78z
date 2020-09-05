1. Word Embedding
1.1 字符级
1.2 词汇级
1.3 通用工具
a. 文本转字符、文本分词
b. 字符、词汇转成词典
c. word2idx, idx2word
d. 生成one-hot vector? 不用nn.embedding提供
e. CBOW (Continuous Bag Of Words) 数据生成。 选定词->预测上下文词汇
f. Skip-gram 数据生成   。上下文->目标词
训练数据是['a','b'], 还是[(['a','b'],c)]?
g. Negative examples
https://rguigoures.github.io/word2vec_pytorch/
h. 训练好的词向量如何导出？ parameters = m.weight.numpy()
https://srijithr.gitlab.io/post/word2vec/
https://discuss.pytorch.org/t/how-to-implement-skip-gram-or-cbow-in-pytorch/47625/3
https://github.com/smafjal/continuous-bag-of-words-pytorch/blob/master/cbow_model_pytorch.py
https://github.com/FraLotito/pytorch-continuous-bag-of-words/blob/master/cbow.py 
j. 除了损失函数，效果如何衡量？
h. tensorbard 显示训练进度？
2. 中文分词

"fullwei:%u dx_0:%d dx_1:%d dx_2:%d dx_3:%d dx_4:%d",
_ugc_data->ugc_result[i].res.full_weight.m_calibrated_model_score,
_ugc_data->ugc_result[i].res.full_weight.m_dx_reference_value[0],
_ugc_data->ugc_result[i].res.full_weight.m_dx_reference_value[1],
_ugc_data->ugc_result[i].res.full_weight.m_dx_reference_value[2],
_ugc_data->ugc_result[i].res.full_weight.m_dx_reference_value[3],
_ugc_data->ugc_result[i].res.full_weight.m_dx_reference_value[4]


ResultQueue_t* final_queue = multiple_queue->GetFinalQueue();

recall_by_mining_before_cut_off

get_scs

ResultQueue_t* queue = p_multi_queue->GetQueueByQuery(QUERY_REALTIME_UGC);

queue->merged_data.bs_out[i]  res
queue->merged_data.real_res = _final_res_num;

queue->merged_data.real_res = _final_res_num;
queue->merged_data.disp_res = _final_res_num;