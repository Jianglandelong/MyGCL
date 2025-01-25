# Actor
python main_test.py -dataset actor -ntrials 10 -sparse 1 -epochs 1000 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.5 -gamma 0.5

# GREET: Run:10 | ACC:36.39+-1.09

# Run:10 | ACC:33.98+-1.02 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.5 -gamma 0.5
# Trial:0 | ACC:35.07
# Trial:1 | ACC:33.88
# Trial:2 | ACC:33.95
# Trial:3 | ACC:33.95
# Trial:4 | ACC:34.54
# Trial:5 | ACC:35.53
# Trial:6 | ACC:33.55
# Trial:7 | ACC:33.36
# Trial:8 | ACC:31.58
# Trial:9 | ACC:34.41

################### alpha small ##################
# Run:10 | ACC:31.80+-1.28 | ACC2:33.13+-1.06 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.0 -beta 0.5 -gamma 0.5
# Run:10 | ACC:32.26+-1.38 | ACC2:33.49+-1.10 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.1 -beta 0.5 -gamma 0.5
# Trial:0 | ACC:31.58 | ACC2:32.50
# Trial:1 | ACC:31.91 | ACC2:34.08
# Trial:2 | ACC:32.57 | ACC2:32.63
# Trial:3 | ACC:32.76 | ACC2:33.16
# Trial:4 | ACC:31.78 | ACC2:33.36
# Trial:5 | ACC:33.49 | ACC2:35.86
# Trial:6 | ACC:33.62 | ACC2:33.62
# Trial:7 | ACC:30.79 | ACC2:33.42
# Trial:8 | ACC:29.61 | ACC2:31.71
# Trial:9 | ACC:34.54 | ACC2:34.54
# Run:10 | ACC:32.04+-1.24 | ACC2:33.52+-0.91 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.3 -beta 0.5 -gamma 0.5
# Run:10 | ACC:32.43+-0.97 | ACC2:33.98+-0.97 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.4 -beta 0.5 -gamma 0.5

################### alpha big #####################
# Run:10 | ACC:32.99+-1.06 | ACC2:33.74+-0.98 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.6 -beta 0.5 -gamma 0.5

# Run:10 | ACC:32.51+-0.93 | ACC2:34.07+-1.05 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.7 -beta 0.5 -gamma 0.5
# Run:10 | ACC:32.80+-0.99 | ACC2:34.45+-0.74 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.8 -beta 0.5 -gamma 0.5
# Run:10 | ACC:32.92+-1.59 | ACC2:34.02+-0.78 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.9 -beta 0.5 -gamma 0.5



################## gamma small ###################
# Run:10 | ACC:33.04+-1.42 | ACC2:34.14+-1.24 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.5 -gamma 0.0
# Trial:5 | ACC:36.38 | ACC2:36.38
# Run:10 | ACC:32.97+-1.00 | ACC2:34.11+-0.77 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.5 -gamma 0.1

################# beta small ######################
# Run:10 | ACC:29.51+-1.50 | ACC2:30.61+-1.02 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.0 -gamma 0.5
# Run:10 | ACC:32.05+-1.02 | ACC2:33.55+-1.11 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.1 -gamma 0.5
# Run:10 | ACC:32.66+-1.45 | ACC2:34.01+-1.13 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.2 -gamma 0.5
# Run:10 | ACC:32.43+-0.82 | ACC2:33.94+-0.98 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.3 -gamma 0.5

#################### gamma zero ###################
# Run:10 | ACC:32.93+-1.02 | ACC2:34.20+-1.01 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.0 -beta 1.0 -gamma 0.0
# Trial:0 | ACC:31.32 | ACC2:33.03
# Trial:1 | ACC:33.82 | ACC2:35.00
# Trial:2 | ACC:33.82 | ACC2:33.82
# Trial:3 | ACC:31.64 | ACC2:32.50
# Trial:4 | ACC:32.11 | ACC2:33.09
# Trial:5 | ACC:34.54 | ACC2:35.86
# Trial:6 | ACC:33.09 | ACC2:34.67
# Trial:7 | ACC:33.09 | ACC2:35.07
# Trial:8 | ACC:32.11 | ACC2:34.34
# Trial:9 | ACC:33.75 | ACC2:34.67

# Run:10 | ACC:32.51+-1.15 | ACC2:33.90+-0.84 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.1 -beta 1.0  -gamma 0.0
# Run:10 | ACC:32.65+-0.76 | ACC2:34.57+-0.76 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.2 -beta 1.0  -gamma 0.0
# Trial:0 | ACC:32.37 | ACC2:33.82
# Trial:1 | ACC:33.75 | ACC2:34.93
# Trial:2 | ACC:32.70 | ACC2:35.79
# Trial:3 | ACC:31.58 | ACC2:33.09
# Trial:4 | ACC:32.11 | ACC2:34.67
# Trial:5 | ACC:33.09 | ACC2:35.13
# Trial:6 | ACC:32.17 | ACC2:34.01
# Trial:7 | ACC:32.11 | ACC2:34.61
# Trial:8 | ACC:32.50 | ACC2:34.21
# Trial:9 | ACC:34.14 | ACC2:35.39
# Run:10 | ACC:32.92+-0.93 | ACC2:34.14+-0.67 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.3 -beta 1.0  -gamma 0.0
# Run:10 | ACC:32.91+-0.84 | ACC2:34.54+-0.74 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.4 -beta 1.0  -gamma 0.0
# Run:10 | ACC:33.30+-1.36 | ACC2:34.25+-0.74 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 1.0  -gamma 0.0

# Run:10 | ACC:33.06+-1.12 | ACC2:34.33+-0.85 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.3 -beta 0.5 -gamma 0.0
# Trial:0 | ACC:34.28 | ACC2:34.28
# Trial:1 | ACC:32.50 | ACC2:34.61
# Trial:2 | ACC:32.43 | ACC2:35.39
# Trial:3 | ACC:31.45 | ACC2:32.57
# Trial:4 | ACC:35.00 | ACC2:35.00
# Trial:5 | ACC:32.63 | ACC2:35.20
# Trial:6 | ACC:34.47 | ACC2:34.61
# Trial:7 | ACC:32.50 | ACC2:33.49
# Trial:8 | ACC:33.42 | ACC2:33.42
# Trial:9 | ACC:31.91 | ACC2:34.74

# Run:10 | ACC:32.73+-1.03 | ACC2:34.00+-0.95 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.7 -beta 1.0  -gamma 0.0
# Run:10 | ACC:32.19+-0.97 | ACC2:33.79+-0.75 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.4 -beta 0.5 -gamma 0.0
# Run:10 | ACC:32.52+-0.84 | ACC2:34.03+-0.62 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.9 -beta 1.0 -gamma 0.0
# Run:10 | ACC:32.72+-1.29 | ACC2:33.99+-0.96 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.5 -gamma 0.0


# Run:10 | ACC:32.76+-1.40 | ACC2:33.95+-1.04 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.1 -beta 0.8 -gamma 0.1
# Trial:5 | ACC:35.53 | ACC2:36.12


# Run:10 | ACC:29.80+-1.01 | ACC2:30.76+-0.5 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.0 -gamma 0.0
# Run:10 | ACC:32.58+-1.20 | ACC2:33.82+-1.06 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.1 -gamma 0.0
# Run:10 | ACC:32.81+-1.47 | ACC2:34.26+-0.96 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 50 -alpha 0.5 -beta 0.2 -gamma 0.0