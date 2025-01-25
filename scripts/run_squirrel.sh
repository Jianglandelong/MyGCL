# Squirrel
python main.py -dataset squirrel -ntrials 8 -sparse 0 -epochs 400 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.5 -gamma 0.0
# -alpha 0.4 -beta 0.5 -gamma 0.5

# Run:8 | ACC:40.50+-0.93 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.5 -gamma 0.5
# Trial:0 | ACC:40.06
# Trial:1 | ACC:41.98
# Trial:2 | ACC:41.40
# Trial:3 | ACC:40.73
# Trial:4 | ACC:40.92
# Trial:5 | ACC:39.00
# Trial:6 | ACC:39.39
# Trial:7 | ACC:40.54

##################### alpha big #####################
# Run:8 | ACC:41.77+-1.29 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.8 -beta 0.5 -gamma 0.5 ***
# Trial:0 | ACC:40.35
# Trial:1 | ACC:41.88
# Trial:2 | ACC:44.19
# Trial:3 | ACC:42.27
# Trial:4 | ACC:43.23
# Trial:5 | ACC:40.63
# Trial:6 | ACC:40.73
# Trial:7 | ACC:40.92
# Run:8 | ACC:40.62+-1.47 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.8 -beta 0.3 -gamma 0.3
# Run:8 | ACC:41.13+-1.33 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.9 -beta 0.5 -gamma 0.5

##################### alpha small #####################
# Run:8 | ACC:39.69+-0.72 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.2 -beta 0.5 -gamma 0.5
# Run:8 | ACC:41.10+-1.67 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.3 -beta 0.5 -gamma 0.5
# Trial:0 | ACC:38.81
# Trial:1 | ACC:40.63
# Trial:2 | ACC:43.52
# Trial:3 | ACC:42.07
# Trial:4 | ACC:41.02
# Trial:5 | ACC:40.25
# Trial:6 | ACC:39.10
# Trial:7 | ACC:43.42
# Run:8 | ACC:40.48+-1.05 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.5

##################### gamma big #####################
# Run:8 | ACC:40.45+-1.93 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.5 -gamma 0.7
# Trial:1 | ACC:42.07
# Trial:3 | ACC:43.61
# Trial:4 | ACC:41.88
# Trial:5 | ACC:41.31
# Run:8 | ACC:40.55+-1.41 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.5 -gamma 0.8
# Trial:1 | ACC:41.31
# Trial:4 | ACC:42.56
# Trial:7 | ACC:42.84

##################### gamma small #####################
# Run:8 | ACC:40.80+-1.60 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.5 -gamma 0.3
# Trial:2 | ACC:43.42
# Trial:4 | ACC:43.52
# Run:8 | ACC:41.59+-1.79 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.5 -gamma 0.4
# Trial:2 | ACC:43.90
# Trial:3 | ACC:43.80
# Trial:4 | ACC:43.32



# Run:10 | ACC:35.73+-1.55 | ACC2:37.77+-0.47 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.0 -beta 1.0 -gamma 0.0