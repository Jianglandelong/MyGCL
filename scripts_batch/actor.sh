#!/bin/bash

# 超参数范围
# alphas=(-0.2 0.0 0.2 0.4 0.6 0.8)
# betas=(0.0 0.4)
# gammas=(-0.2 0.0 0.2 0.4 0.6 0.8)
# alphas=(0.3 0.4 0.5 0.6 0.7 0.8)
# betas=(1.0)
# gammas=(0.0)

# # 遍历所有组合
# for alpha in "${alphas[@]}"; do
#   for beta in "${betas[@]}"; do
#     for gamma in "${gammas[@]}"; do
#       python main_test.py -dataset actor -ntrials 8 -sparse 1 -epochs 1000 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -full_log 0 >> log/actor.log
#       echo "" >> log/actor.log
#     done
#   done
# done

# echo "########################## emb_dim 256 ##########################" >> log/actor.log
# for alpha in "${alphas[@]}"; do
#   for beta in "${betas[@]}"; do
#     for gamma in "${gammas[@]}"; do
#       python main_test.py -dataset actor -ntrials 8 -sparse 1 -epochs 1000 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha ${alpha} -beta ${beta} -gamma ${gamma} -emb_dim 256 -full_log 0 >> log/actor.log
#       echo "" >> log/actor.log
#     done
#   done
# done

dropedge_rates=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
for dropedge_rate in "${dropedge_rates[@]}"; do
  python main_test.py -dataset actor -ntrials 8 -sparse 1 -epochs 1000 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 ${dropedge_rate} -dropedge_rate_2 ${dropedge_rate} -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 1.0 -gamma 0.0 -full_log 0 >> log/actor.log
  echo "" >> log/actor.log
done

# enc_layers=(1 2)
# proj_layers=(1 2 3 4 5)
# for enc_layer in "${enc_layers[@]}"; do
#   for proj_layer in "${proj_layers[@]}"; do
#     python main_tmp.py -dataset actor -ntrials 8 -sparse 1 -epochs 1000 -cl_batch_size 0 -nlayers_enc ${enc_layer} -nlayers_proj ${proj_layer} -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 1.0 -gamma 0.0 -full_log 0 >> log/actor.log
#     echo "" >> log/actor.log
#   done
# done