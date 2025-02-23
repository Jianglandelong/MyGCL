#!/bin/bash

# dropedge_rates=(0.0 0.2 0.4 0.6 0.8 0.9 0.95 0.99)
# for dropedge_rate in "${dropedge_rates[@]}"; do
#   python main_pubmed.py -dataset physics -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 2000 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 ${dropedge_rate} -dropedge_rate_2 ${dropedge_rate} -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.0 -gamma 0.0 -full_log 0 >> log/coauthor_physics.log
#   echo "" >> log/coauthor_physics.log
# done

# dropnode_rates=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# maskfeat_rates=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# maskfeat_rates=(0.3 0.4)
# dropnode_rates=(0.0 0.1 0.2 0.3 0.4)
# for maskfeat_rate in "${maskfeat_rates[@]}"; do
#   for dropnode_rate in "${dropnode_rates[@]}"; do
#     python main_pubmed.py -dataset physics -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 2000 -nlayers_proj 1 -maskfeat_rate_1 ${maskfeat_rate} -maskfeat_rate_2 ${maskfeat_rate} -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 ${dropnode_rate} -dropnode_rate_2 ${dropnode_rate} -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.0 -gamma 0.0 -emb_dim 512 -full_log 0 >> log/coauthor_physics.log
#     echo "" >> log/coauthor_physics.log
#   done
# done

# betas=(-0.5 -0.4 -0.3 -0.2 -0.1)
# betas=(0.1 0.2 0.3 0.4 0.5)
# betas=(0.0 0.1)
# gammas=(0.0 0.1)
# maskfeat_rates=(0.1 0.2 0.3)
# betas=(0.2 0.3)
# gammas=(0.0 0.1 0.2)
# for beta in "${betas[@]}"; do
#   for gamma in "${gammas[@]}"; do
#     for maskfeat_rate in "${maskfeat_rates[@]}"; do
#       python main_pubmed.py -dataset physics -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 2000 -nlayers_proj 1 -maskfeat_rate_1 ${maskfeat_rate} -maskfeat_rate_2 ${maskfeat_rate} -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.2 -dropnode_rate_2 0.2 -lr 0.001 -eval_freq 50 -alpha 1.0 -beta ${beta} -gamma ${gamma} -emb_dim 512 -full_log 0 >> log/coauthor_physics.log
#       echo "" >> log/coauthor_physics.log
#     done
#   done
# done

python main_pubmed.py -dataset physics -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 2000 -nlayers_proj 1 -maskfeat_rate_1 0.3 -maskfeat_rate_2 0.3 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.2 -dropnode_rate_2 0.2 -lr 0.001 -eval_freq 50 -alpha 1.0 -beta 0.2 -gamma 0.1 -emb_dim 1024 -full_log 0 >> log/coauthor_physics.log
echo "" >> log/coauthor_physics.log

python main_pubmed.py -dataset physics -ntrials 8 -sparse 1 -epochs 800 -cl_batch_size 2000 -nlayers_proj 1 -maskfeat_rate_1 0.3 -maskfeat_rate_2 0.3 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.2 -dropnode_rate_2 0.2 -lr 0.001 -eval_freq 50 -alpha 1.0 -beta 0.2 -gamma 0.1 -emb_dim 1536 -full_log 0 >> log/coauthor_physics.log
echo "" >> log/coauthor_physics.log