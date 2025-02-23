#!/bin/bash

# 超参数范围
# alphas=(-0.2 0.0 0.2 0.4 0.6 0.8)
# alphas=(2.0 2.2 2.4 2.6 2.8)
# betas=(0.4)
# gammas=(0.2 0.4)

# # 遍历所有组合
# for alpha in "${alphas[@]}"; do
#   for beta in "${betas[@]}"; do
#     for gamma in "${gammas[@]}"; do
#       python main_test.py -dataset squirrel -ntrials 8 -sparse 0 -epochs 400 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -full_log 0 >> log/squirrel.log
#       echo "" >> log/squirrel.log
#     done
#   done
# done

# dropedge_rates=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# dropnode_rates=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# dropnode_rates=(0.95 0.99)
# maskfeat_rates=(0.0 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
# maskfeat_rates=(0.1 0.2)
# dropnode_rates=(0.0 0.1)
# dropedge_rates=(0.2 0.3 0.4)
# for dropedge_rate in "${dropedge_rates[@]}"; do
#   python main_test.py -dataset squirrel -ntrials 8 -sparse 0 -epochs 400 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 ${dropedge_rate} -dropedge_rate_2 ${dropedge_rate} -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.8 -beta 0.4 -gamma 0.0 -emb_dim 512 -full_log 0 >> log/squirrel.log
#   echo "" >> log/squirrel.log
# done

# gammas=(0.0 0.2 0.4 0.6 0.8)
# for gamma in "${gammas[@]}"; do
#   python main_test.py -dataset squirrel -ntrials 8 -sparse 0 -epochs 400 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.8 -beta 0.4 -gamma ${gamma} -emb_dim 256 -full_log 0 >> log/squirrel.log
#   echo "" >> log/squirrel.log
# done

python main_test.py -dataset squirrel -ntrials 8 -sparse 0 -epochs 400 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.8 -beta 0.4 -gamma 0.0 -emb_dim 2048 -full_log 0 >> log/squirrel.log
echo "" >> log/squirrel.log