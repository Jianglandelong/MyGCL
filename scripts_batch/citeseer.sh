#!/bin/bash

# # 超参数范围
# alphas=(1.0)
# betas=(-0.2 -0.1 0.0 0.1 0.2)
# gammas=(-0.2 -0.1 0.0 0.1 0.2)

# # 遍历所有组合
# for alpha in "${alphas[@]}"; do
#   for beta in "${betas[@]}"; do
#     for gamma in "${gammas[@]}"; do
#       python main_test.py -dataset citeseer -ntrials 8 -sparse 0 -epochs 400 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -full_log 0 >> log/citeseer.log
#       echo "" >> log/citeseer.log
#     done
#   done
# done

# 超参数范围
dropedge_rates=(0.3 0.4 0.5 0.6 0.7 0.8)

for i in "${!dropedge_rates[@]}"; do
  dropedge_rate=${dropedge_rates[$i]}
  python main_test.py -dataset citeseer -ntrials 8 -sparse 0 -epochs 400 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 ${dropedge_rate} -dropedge_rate_2 ${dropedge_rate} -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256 -full_log 0 >> log/citeseer.log
done

# maskfeat_rates=(0.6 0.7 0.8)
# dropedge_rates=(0.3 0.4 0.5 0.6 0.7)
maskfeat_rates=(0.6 0.7)
dropedge_rates=(0.6 0.7)

# 遍历所有组合
for maskfeat_rate in "${maskfeat_rates[@]}"; do
  for dropedge_rate in "${dropedge_rates[@]}"; do
    python main_test.py -dataset citeseer -ntrials 8 -sparse 0 -epochs 400 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 ${maskfeat_rate} -maskfeat_rate_2 ${maskfeat_rate} -dropedge_rate_1 ${dropedge_rate} -dropedge_rate_2 ${dropedge_rate} -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256 -full_log 0 >> log/citeseer.log
    echo "" >> log/citeseer.log
  done
done