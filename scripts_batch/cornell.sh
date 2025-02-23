#!/bin/bash

# 超参数范围
# alphas=(-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5)
# betas=(1.0)
# gammas=(-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5)
# alphas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2)
# betas=(1.0)
# gammas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2)

# # 遍历所有组合
# for alpha in "${alphas[@]}"; do
#   for beta in "${betas[@]}"; do
#     for gamma in "${gammas[@]}"; do
#       if (( $(echo "$alpha <= 0.5" | bc -l) && $(echo "$gamma <= 0.5" | bc -l) )); then
#         continue
#       fi
#       python main_test.py -dataset cornell -ntrials 8 -sparse 0 -epochs 400 -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -full_log 0 >> log/cornell.log
#       echo "" >> log/cornell.log
#     done
#   done
# done

# alphas=(0.2 0.3 0.4 0.5 0.6)
# gammas=(-0.1 0.0 0.1)
alphas=(0.5 0.6 0.7 0.8 0.9 1.0)
gammas=(0.0 0.1 0.3 0.4 0.5)
for alpha in "${alphas[@]}"; do
  for gamma in "${gammas[@]}"; do
    python main_test.py -dataset cornell -ntrials 8 -sparse 0 -epochs 400 -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.7 -dropedge_rate_2 0.7 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha ${alpha} -beta 1.0 -gamma ${gamma} -emb_dim 1024 -full_log 0 >> log/cornell.log
    echo "" >> log/cornell.log
  done
done 

# 超参数范围
# maskfeat_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
# dropedge_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
# dropnode_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)

# for i in "${!maskfeat_rates[@]}"; do
#   maskfeat_rate=${maskfeat_rates[$i]}
#   python main_test.py -dataset cornell -ntrials 8 -sparse 0 -epochs 400 -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 ${maskfeat_rate} -maskfeat_rate_2 ${maskfeat_rate} -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.3 -beta 1.0 -gamma -0.1 -full_log 0 >> log/cornell.log
#   echo "" >> log/cornell.log
# done

# for i in "${!dropedge_rates[@]}"; do
#   dropedge_rate=${dropedge_rates[$i]}
#   python main_test.py -dataset cornell -ntrials 8 -sparse 0 -epochs 400 -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.0 -maskfeat_rate_2 0.0 -dropedge_rate_1 ${dropedge_rate} -dropedge_rate_2 ${dropedge_rate} -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.3 -beta 1.0 -gamma -0.1 -full_log 0 >> log/cornell.log
#   echo "" >> log/cornell.log
# done

# for i in "${!dropnode_rates[@]}"; do
#   dropnode_rate=${dropnode_rates[$i]}
#   python main_test.py -dataset cornell -ntrials 8 -sparse 0 -epochs 400 -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.0 -maskfeat_rate_2 0.0 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 ${dropnode_rate} -dropnode_rate_2 ${dropnode_rate} -lr 0.001 -eval_freq 10 -alpha 0.3 -beta 1.0 -gamma -0.1 -full_log 0 >> log/cornell.log
#   echo "" >> log/cornell.log
# done

# maskfeat_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
# dropedge_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)

# # 遍历所有组合
# for maskfeat_rate in "${maskfeat_rates[@]}"; do
#   for dropedge_rate in "${dropedge_rates[@]}"; do
#     python main_test.py -dataset cornell -ntrials 8 -sparse 0 -epochs 400 -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 ${maskfeat_rate} -maskfeat_rate_2 ${maskfeat_rate} -dropedge_rate_1 ${dropedge_rate} -dropedge_rate_2 ${dropedge_rate} -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.3 -beta 1.0 -gamma -0.1 -full_log 0 >> log/cornell.log
#   echo "" >> log/cornell.log
#   done
# done