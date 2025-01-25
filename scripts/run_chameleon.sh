# Chameleon
python main.py -dataset chameleon -ntrials 8 -sparse 0 -epochs 500 -cl_batch_size 0 -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.5 -gamma 0.5
# citeseer: 0.2, 0.5, 0.5

# Run:8 | ACC:55.62+-2.21 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5

################### gamma big ######################
# Run:8 | ACC:56.74+-3.06 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.7
# Run:8 | ACC:59.32+-2.60 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.9
# Trial:1 | ACC:62.28
# Run:8 | ACC:58.25+-1.2 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.3 -gamma 0.6
# Run:8 | ACC:59.27+-1.49 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.3 -gamma 0.7
# Run:8 | ACC:59.43+-2.71| -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.3 -gamma 0.8
# Trial:1 | ACC:63.38
# Trial:3 | ACC:62.06
# Trial:4 | ACC:62.72
# Run:8 | ACC:59.05+-2.33 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.3 -gamma 0.9
# Run:8 | ACC:58.69+-2.40 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.3 -gamma 1.0
# Run:8 | ACC:58.44+-1.91 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.3 -gamma 1.1

################### gamma small #################
# Run:8 | ACC:57.98+-1.35 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.3 -gamma 0.0
# Run:8 | ACC:58.88+-2.90 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.3 -gamma 0.1
# Run:8 | ACC:59.87+-1.53 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.3 -beta 0.3 -gamma 0.2
# Run:8 | ACC:60.22+-2.77 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.4 -beta 0.4 -gamma 0.3 ***
# Trial:0 | ACC:59.43
# Trial:1 | ACC:66.01
# Trial:2 | ACC:58.11
# Trial:3 | ACC:56.14
# Trial:4 | ACC:59.87
# Trial:5 | ACC:62.50
# Trial:6 | ACC:60.31
# Trial:7 | ACC:59.43
# Run:8 | ACC:58.85+-2.38 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.4
# Run:8 | ACC:59.87+-2.69 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.6 -beta 0.6 -gamma 0.5
# Run:8 | ACC:59.46+-2.28 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.7 -beta 0.7 -gamma 0.6
# Trial:1 | ACC:64.91

################## alpha small ###################
# Run:8 | ACC:52.93+-2.75 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.0 -beta 0.5 -gamma 0.5
# Run:8 | ACC:56.22+-2.30 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.1 -beta 0.5 -gamma 0.5