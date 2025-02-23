#!/bin/bash

# 超参数范围
# alphas=(1.0)
# betas=(-0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5)
# gammas=(-0.1 0.0 0.1 0.2 0.3 0.4 0.5)
# betas=(0.0 0.1 0.2 0.3)
# gammas=(-0.1 0.0 0.1 0.2 0.3 0.4)

# # 遍历所有组合
# for alpha in "${alphas[@]}"; do
#   for beta in "${betas[@]}"; do
#     for gamma in "${gammas[@]}"; do
#       python main_pubmed.py -dataset pubmed -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 5000 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -emb_dim 256 -full_log 0 >> log/pubmed.log
#       echo "" >> log/pubmed.log
#     done
#   done
# done

# dropnode_rates=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# maskfeat_rates=(0.0 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# for maskfeat_rate in "${maskfeat_rates[@]}"; do
#   python main_pubmed.py -dataset pubmed -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 5000 -nlayers_proj 2 -maskfeat_rate_1 ${maskfeat_rate} -maskfeat_rate_2 ${maskfeat_rate} -dropedge_rate_1 0.7 -dropedge_rate_2 0.7 -dropnode_rate_1 0.2 -dropnode_rate_2 0.2 -lr 0.001 -eval_freq 20 -alpha 1.0 -beta 0.1 -gamma 0.4 -emb_dim 256 -full_log 0 >> log/pubmed.log
#   echo "" >> log/pubmed.log
# done

# # 超参数按下标相同的进行组合
alphas=(1.1)
betas=(-0.1 0.0 0.1 0.2)
gammas=(-0.2 -0.1 0.0 0.1 0.2)
# alphas=(1.0 1.1 1.2 1.3 1.4)
# betas=(0.1)
# gammas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)

# 遍历所有组合
for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    for gamma in "${gammas[@]}"; do
      python main_pubmed.py -dataset pubmed -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 5000 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.7 -dropedge_rate_2 0.7 -dropnode_rate_1 0.2 -dropnode_rate_2 0.2 -lr 0.001 -eval_freq 20 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -emb_dim 512 -full_log 0 >> log/pubmed.log
      echo "" >> log/pubmed.log
    done
  done
done