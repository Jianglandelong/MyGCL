#!/bin/bash

# 超参数范围
alphas=(1.0)
betas=(-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5)
gammas=(-0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5)

# 遍历所有组合
for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    for gamma in "${gammas[@]}"; do
      python main_test.py -dataset pubmed -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 5000 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -lr 0.001 -eval_freq 20 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -full_log 0 >> log/pubmed.log
      echo "" >> log/pubmed.log
    done
  done
done

# # 超参数按下标相同的进行组合
# alphas=(1.0)
# betas=(0.0 0.0 0.0 0.1 0.1 0.1 0.2 0.2 0.2 0.3 0.3 0.3 0.3 0.3 0.3 0.4 0.4 0.4 0.4 0.4 0.4 0.5 0.5 0.5 0.5 0.5 0.5)
# gammas=(-0.3 -0.4 -0.5 -0.3 -0.4 -0.5 -0.3 -0.4 -0.5 0.0 -0.1 -0.2 -0.3 -0.4 -0.5 0.0 -0.1 -0.2 -0.3 -0.4 -0.5 0.0 -0.1 -0.2 -0.3 -0.4 -0.5)

# # 遍历所有组合
# for alpha in "${alphas[@]}"; do
#   for i in "${!betas[@]}"; do
#     beta=${betas[$i]}
#     gamma=${gammas[$i]}
#     python main_test.py -dataset pubmed -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 5000 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -lr 0.001 -eval_freq 20 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -full_log 0 >> log/pubmed.log
#       echo "" >> log/pubmed.log
#   done
# done