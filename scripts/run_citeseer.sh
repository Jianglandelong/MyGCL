# CiteSeer
python main_test.py -dataset citeseer -ntrials 8 -sparse 0 -epochs 400 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0

# GREET: Run:10 | ACC:73.18+-0.79

# Run:8 | ACC:63.84+-2.29 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0
# Run:8 | ACC:65.52+-1.84 | -nlayers_proj 2 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 
# Run:8 | ACC:65.85+-2.45 | -nlayers_proj 2 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0
# Run:8 | ACC:67.39+-2.12 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0
# Trial:1 | ACC:70.10
# Trial:2 | ACC:70.40
  ###################### emb dim ##################
  # Run:8 | ACC:67.40+-1.70 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256
  # Run:8 | ACC:67.38+-1.94 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256

# Run:8 | ACC:67.60+-1.45 | -nlayers_proj 2 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0
# Run:8 | ACC:67.86+-2.25 | -nlayers_proj 2 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 ***
# Trial:2 | ACC:71.00
# Trial:6 | ACC:70.30
# Run:8 | ACC:67.89+-2.13 | -nlayers_proj 2 -maskfeat_rate_1 0.6 -maskfeat_rate_2 0.6 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0
# Trial:1 | ACC:70.30
# Trial:2 | ACC:71.10
# Run:8 | ACC:66.91+-2.26 | -nlayers_proj 2 -maskfeat_rate_1 0.6 -maskfeat_rate_2 0.6 -dropedge_rate_1 0.6 -dropedge_rate_2 0.6 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0
############## pin maskfeat_rate to 0.6 #################
# Run:8 | ACC:66.30+-1.97 | -nlayers_proj 2 -maskfeat_rate_1 0.6 -maskfeat_rate_2 0.6 -dropedge_rate_1 0.3 -dropedge_rate_2 0.3 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0
# Run:8 | ACC:68.09+-1.69 | -nlayers_proj 2 -maskfeat_rate_1 0.6 -maskfeat_rate_2 0.6 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0
# Trial:0 | ACC:68.30
# Trial:1 | ACC:68.00
# Trial:2 | ACC:70.20
# Trial:3 | ACC:68.50
# Trial:4 | ACC:68.50
# Trial:5 | ACC:64.60
# Trial:6 | ACC:70.00
# Trial:7 | ACC:66.60
# Run:8 | ACC:67.06+-1.48 | -nlayers_proj 2 -maskfeat_rate_1 0.6 -maskfeat_rate_2 0.6 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256

################## beta big ####################
# Run:8 | ACC:66.78+-2.36 | -nlayers_proj 2 -maskfeat_rate_1 0.6 -maskfeat_rate_2 0.6 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.1 -gamma 0.0
# Run:8 | ACC:66.24+-2.32 | -nlayers_proj 2 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.2 -gamma 0.0
# Run:8 | ACC:65.27+-1.50 | -nlayers_proj 2 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -alpha 0.8 -beta 0.2 -gamma 0.0 

################# gamma big ######################
# Run:8 | ACC:62.44+-2.71 | -nlayers_proj 2 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.2