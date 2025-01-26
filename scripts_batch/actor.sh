#!/bin/bash

# 超参数范围
alphas=(-0.2 0.0 0.2 0.4 0.6 0.8)
betas=(0.0 0.4)
gammas=(-0.2 0.0 0.2 0.4 0.6 0.8)

# 遍历所有组合
for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    for gamma in "${gammas[@]}"; do
      python main_test.py -dataset actor -ntrials 10 -sparse 1 -epochs 1000 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -emb_dim 256 -full_log 0 >> log/texas.log
      echo "" >> log/texas.log
    done
  done
done