#!/bin/bash

# 超参数范围
alphas=(1.0)
betas=(-0.2 -0.1 0.0 0.1 0.2)
gammas=(-0.2 -0.1 0.0 0.1 0.2)

# 遍历所有组合
for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    for gamma in "${gammas[@]}"; do
      python main_test.py -dataset citeseer -ntrials 8 -sparse 0 -epochs 400 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -full_log 0 >> log/citeseer.log
      echo "" >> log/citeseer.log
    done
  done
done