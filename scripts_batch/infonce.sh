#!/bin/bash

# Done
# cora
# citeseer
# cornell
# texas
# wisconsin
# actor
#ama_photo
#chameleon
#ama_photo
#coauthor_cs(fail)
#ama_computers(fail)
#squirrel
#coauthor_physics(fail)
#pubmed(fail)
#wiki_cs(fail)

# 注意执行main_infonce.py，写入infonce.log，
echo "#coauthor_physics" >> log/infonce.log
python main_infonce.py -dataset physics -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 2000 -nlayers_proj 1 -maskfeat_rate_1 0.3 -maskfeat_rate_2 0.3 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.2 -dropnode_rate_2 0.2 -lr 0.001 -eval_freq 50 -alpha 1.0 -beta 0.2 -gamma 0.1 -emb_dim 128 -full_log 0 >> log/infonce.log
echo "" >> log/infonce.log

echo "#pubmed" >> log/infonce.log
python main_infonce.py -dataset pubmed -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 5000 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.7 -dropedge_rate_2 0.7 -dropnode_rate_1 0.2 -dropnode_rate_2 0.2 -lr 0.001 -eval_freq 20 -alpha 1.0 -beta 0.1 -gamma 0.0 -emb_dim 128 -full_log 0 >> log/infonce.log
echo "" >> log/infonce.log

echo "#wiki_cs" >> log/infonce.log
python main_infonce.py -dataset wikics -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 3000 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.7 -dropedge_rate_2 0.7 -dropnode_rate_1 0.1 -dropnode_rate_2 0.1 -lr 0.001 -eval_freq 50 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 128 -full_log 0 >> log/infonce.log
echo "" >> log/infonce.log