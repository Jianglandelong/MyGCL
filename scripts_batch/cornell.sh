#!/bin/bash

# 超参数范围
alphas=(-0.2 -0.1 0.0 0.1 0.2)
betas=(1.0)
gammas=(-0.2 -0.1 0.0 0.1 0.2)

# 遍历所有组合
for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    for gamma in "${gammas[@]}"; do
      python main_test.py -dataset cornell -ntrials 8 -sparse 0 -epochs 400 -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -emb_dim 256 -full_log 0 >> log/cornell.log
      echo "" >> log/cornell.log
    done
  done
done