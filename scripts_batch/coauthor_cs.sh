#!/bin/bash

# # 超参数范围
# dropedge_rates=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# dropedge_rates=(0.0 0.4)
# dropnode_rates=(0.0 0.2 0.4 0.6 0.8)

# for dropedge_rate in "${dropedge_rates[@]}"; do
#   for dropnode_rate in "${dropnode_rates[@]}"; do
#     python main_pubmed.py -dataset cs -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 ${dropedge_rate} -dropedge_rate_2 ${dropedge_rate} -dropnode_rate_1 ${dropnode_rate} -dropnode_rate_2 ${dropnode_rate} -lr 0.001 -eval_freq 50 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256 -full_log 0 >> log/coauthor_cs.log
#     echo "" >> log/coauthor_cs.log
#   done
# done

# maskfeat_rates=(0.0 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
maskfeat_rates=(0.2 0.6 0.7)
for maskfeat_rate in "${maskfeat_rates[@]}"; do
  python main_pubmed.py -dataset cs -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -maskfeat_rate_1 ${maskfeat_rate} -maskfeat_rate_2 ${maskfeat_rate} -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.1 -dropnode_rate_2 0.1 -lr 0.001 -eval_freq 50 -alpha 1.0 -beta -0.1 -gamma 0.3 -emb_dim 2048 -full_log 0 >> log/coauthor_cs.log
  echo "" >> log/coauthor_cs.log
done

# maskfeat_rates=(0.0 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
# betas=(-0.1 -0.2 -0.3 -0.4 -0.5)
# gammas=(-0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5)

# for gamma in "${gammas[@]}"; do
#   python main_pubmed.py -dataset cs -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -lr 0.001 -eval_freq 50 -alpha 1.0 -beta -0.1 -gamma ${gamma} -emb_dim 512 -full_log 0 >> log/coauthor_cs.log
#   echo "" >> log/coauthor_cs.log
# done

# python main_pubmed.py -dataset cs -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -lr 0.001 -eval_freq 50 -alpha 1.0 -beta 0.0 -gamma 0.0 -full_log 0 -emb_dim 256 >> log/coauthor_cs.log
# echo "" >> log/coauthor_cs.log

# echo "############################## emb_dim=512 ###############################" >> log/coauthor_cs.log

# python main_pubmed.py -dataset cs -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -lr 0.001 -eval_freq 50 -alpha 1.0 -beta 0.0 -gamma 0.0 -full_log 0 -emb_dim 512 >> log/coauthor_cs.log
# echo "" >> log/coauthor_cs.log