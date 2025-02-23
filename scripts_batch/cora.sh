#!/bin/bash

# 超参数范围
alphas=(1.0)
# betas=(-0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4)
# gammas=(-0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4)
betas=(-0.1 0.0 0.1 0.2)
gammas=(-0.6 -0.5 0.5 -0.6)

# 遍历所有组合
for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    for gamma in "${gammas[@]}"; do
      # if (( $(echo "$beta >= -0.2" | bc -l) && $(echo "$beta <= 0.2" | bc -l) && $(echo "$gamma >= -0.2" | bc -l) && $(echo "$gamma <= 0.2" | bc -l) )); then
      #   continue
      # fi
      python main_test.py -dataset cora -ntrials 8 -sparse 0 -epochs 400 -cl_batch_size 0 -nlayers_proj 1 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0005 -eval_freq 20 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -emb_dim 256 -full_log 0 >> log/cora.log
      echo "" >> log/cora.log
    done
  done
done