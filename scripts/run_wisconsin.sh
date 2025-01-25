# Wisconsin
python main.py -dataset wisconsin -ntrials 8 -sparse 0 -epochs 400 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.0 -beta 0.5 -gamma 0.4

# GREET: Run:10 | ACC:84.12+-4.15

# Run:8 | ACC:73.04+-4.13 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.3 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5
# Run:8 | ACC:74.75+-5.84 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5 
# Trial:1 | ACC:86.27
# Run:8 | ACC:72.55+-5.88 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.5 -gamma 0.5 -emb_dim 256
# Run:8 | ACC:73.53+-6.43 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.5 -gamma 0.5 -emb_dim 512

################# alpha small ###################
# Run:8 | ACC:71.57+-4.80 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.0 -beta 0.5 -gamma 0.0
# Run:8 | ACC:74.26+-6.16 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.3 -beta 0.5 -gamma 0.5
# Run:8 | ACC:74.75+-7.56 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.5 ***
# Trial:0 | ACC:74.51
# Trial:1 | ACC:86.27
# Trial:2 | ACC:62.75
# Trial:3 | ACC:66.67
# Trial:4 | ACC:70.59
# Trial:5 | ACC:76.47
# Trial:6 | ACC:84.31
# Trial:7 | ACC:76.47
  ################## emb dim #####################
  # Run:8 | ACC:72.06+-4.88 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.5 -emb_dim 256
  # Run:8 | ACC:73.77+-3.24 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.5 -emb_dim 512
  # Run:8 | ACC:72.55+-7.84 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.5 -emb_dim 512
  # Trial:1 | ACC:90.20 ***
  # Run:8 | ACC:75.25+-4.70 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.5 -emb_dim 512 ***
  # Trial:0 | ACC:74.51
  # Trial:1 | ACC:82.35
  # Trial:2 | ACC:70.59
  # Trial:3 | ACC:68.63
  # Trial:4 | ACC:76.47
  # Trial:5 | ACC:74.51
  # Trial:6 | ACC:82.35
  # Trial:7 | ACC:72.55
  # Run:8 | ACC:73.53+-5.55 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.5 -emb_dim 512

############### alpha big ########################
  ################## emb dim #######################
  # Run:8 | ACC:71.08+-4.47 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.6 -beta 0.5 -gamma 0.5 -emb_dim 512

############## gamma big ######################
# Run:8 | ACC:74.26+-5.75 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.6
# Trial:1 | ACC:82.35
# Trial:6 | ACC:82.35
# Run:8 | ACC:73.28+-6.50 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.4 -gamma 0.5
# Run:8 | ACC:71.57+-6.1 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.7
# Run:8 | ACC:72.79+-5.58 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.8

############### gamma small ####################
# Run:8 | ACC:70.59+-5.00 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.2
# Run:8 | ACC:71.32+-4.38 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.4

############### beta big ######################
# Run:8 | ACC:72.79+-6.16 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.6 -gamma 0.5
# Trial:1 | ACC:82.35
# Run:8 | ACC:73.53+-7.40 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.4 ***
# Trial:1 | ACC:86.27
# Trial:6 | ACC:84.31
# Run:8 | ACC:73.28+-7.53| -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.7 -gamma 0.5
# Trial:6 | ACC:84.31
# Run:8 | ACC:70.34+-3.97 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.8 -gamma 0.5

################ beta small ###################
# Run:8 | ACC:72.06+-5.35 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.4 -gamma 0.5

################# pin alpha to 0.4 #################
  ################### gamma small ##################
  # Run:8 | ACC:71.08+-5.70 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.3
    ################## emb dim #######################
    # Run:8 | ACC:71.08+-4.68 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.4 -emb_dim 512
    # Run:8 | ACC:71.08+-4.68 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.4 -emb_dim 512

  ################### gamma big ####################
  # Run:8 | ACC:71.81+-5.45 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.6
    #################### emb dim ######################
    # Run:8 | ACC:74.26+-5.50 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.6 -emb_dim 512
    # Run:8 | ACC:73.53+-5.80 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.5 -gamma 0.7 -emb_dim 512

  ################### beta big #####################
  # Run:8 | ACC:70.83+-3.97 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.6 -gamma 0.5
    ###################### emb dim ####################
    # Run:8 | ACC:72.79+-5.84 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.6 -gamma 0.5 -emb_dim 512
    # Run:8 | ACC:71.08+-6.48 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.7 -gamma 0.5 -emb_dim 512
  
  #################### beta smalll #####################
    ##################### emb dim ########################
    # Run:8 | ACC:73.04+-4.13 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.4 -beta 0.4 -gamma 0.5 -emb_dim 512

################  pin beta #####################
# Run:8 | ACC:71.57+-4.80 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.0 -beta 0.5 -gamma 0.0
# Run:8 | ACC:73.53+-5.72 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.1 -beta 0.5 -gamma 0.0
# Trial:1 | ACC:86.27
# Run:8 | ACC:74.26+-5.75 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.2 -beta 0.5 -gamma 0.0
# Trial:1 | ACC:88.24
# Run:8 | ACC:71.32+-6.12 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.3 -beta 0.5 -gamma 0.0

# Run:8 | ACC:74.26+-6.31 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.0 -beta 0.5 -gamma 0.1
# Trial:1 | ACC:90.20
# Run:8 | ACC:72.79+-6.54 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.0 -beta 0.5 -gamma 0.2
# Trial:1 | ACC:88.24
# Run:8 | ACC:71.81+-6.20 | python main.py -dataset wisconsin -ntrials 8 -sparse 0 -epochs 400 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.0 -beta 0.5 -gamma 0.3
