#!/bin/bash

# # 超参数范围
# dropedge_rates=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)

# # 遍历所有组合
# for dropedge_rate in "${dropedge_rates[@]}"; do
#   python main_test.py -dataset photo -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 ${dropedge_rate} -dropedge_rate_2 ${dropedge_rate} -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 20 -alpha 0.5 -beta 0.0 -gamma 0 -full_log 0 >> log/ama_photo.log
#   echo "" >> log/ama_photo.log
# done

# maskfeat_rates=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
maskfeat_rates=(0.1 0.2 0.3)

# 遍历所有组合
for maskfeat_rate in "${maskfeat_rates[@]}"; do
  python main_test.py -dataset photo -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -maskfeat_rate_1 ${maskfeat_rate} -maskfeat_rate_2 ${maskfeat_rate} -dropedge_rate_1 0.99 -dropedge_rate_2 0.99 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 20 -alpha 0.5 -beta 0.0 -gamma 0 -emb_dim 512 -full_log 0 >> log/ama_photo.log
  echo "" >> log/ama_photo.log
done

# python main_test.py -dataset photo -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.99 -dropedge_rate_2 0.99 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 20 -alpha 0.5 -beta 0.0 -gamma 0 -full_log 0 >> log/ama_photo.log
# echo "" >> log/ama_photo.log

# python main_test.py -dataset photo -ntrials 8 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 1.0 -dropedge_rate_2 1.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 20 -alpha 0.5 -beta 0.0 -gamma 0 -full_log 0 >> log/ama_photo.log
# echo "" >> log/ama_photo.log