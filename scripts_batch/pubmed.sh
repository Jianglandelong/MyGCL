#!/bin/bash

# 超参数范围
alphas=(1.0)
betas=(-0.2 -0.1 0.0 0.1 0.2)
gammas=(-0.2 -0.1 0.0 0.1 0.2)

# 遍历所有组合
for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    for gamma in "${gammas[@]}"; do
      python main_test.py -dataset pubmed -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 5000 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -lr 0.001 -eval_freq 20 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -full_log 0 >> log/pubmed.log
      echo "" >> log/pubmed.log
    done
  done
done