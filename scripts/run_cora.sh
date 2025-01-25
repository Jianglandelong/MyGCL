# Cora
python main.py -dataset cora -ntrials 8 -sparse 0 -epochs 400 -cl_batch_size 0 -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 1.0 -beta 0.2 -gamma 0.0 -emb_dim 256 -nlayers_enc 3

# time: 1382.82 seconds 8 trials
# Trial:0 | ACC:79.30
# Trial:1 | ACC:79.50
# Trial:2 | ACC:80.60
# Trial:3 | ACC:81.20
# Trial:4 | ACC:78.40
# Trial:5 | ACC:80.70

# [FINAL RESULT] Dataset:cora | Run:6 | ACC:79.95+-0.96 | -nlayers_proj 1

# Run:8 | ACC:76.79+-1.56 | -nlayers_proj 2
# Run:8 | ACC:79.05+-1.04 | -nlayrs_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4
# Run:8 | ACC:78.89+-0.93 | -nlayers_proj 1 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.1 -dropnode_rate_2 0.1
# Run:8 | ACC:78.30+-1.34 | -nlayers_proj 1 -maskfeat_rate_1 0.0 -maskfeat_rate_2 0.0 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0
# Run:8 | ACC:79.41+-1.51 | -nlayers_proj 1 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0
# Run:8 | ACC:79.16+-1.17 | -nlayers_proj 1 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0
# Run:8 | ACC:80.16+-0.39 | -nlayers_proj 1 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0005 -eval_freq 20 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256 ***
# Run:8 | ACC:79.08+-0.89 | -nlayers_proj 1 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0005 -eval_freq 20 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 512
# Run:8 | ACC:79.04+-0.95 | -nlayers_proj 1 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0005 -eval_freq 20 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256 -nlayers_enc 3
# Run:8 | ACC:79.71+-0.91 | -nlayers_proj 1 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0005 -eval_freq 20 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256
# Run:8 | ACC:79.08+-0.89 | -nlayers_proj 1 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0005 -eval_freq 20 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 512
# Run:8 | ACC:77.74+-0.73 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0005 -eval_freq 20 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256
# Run:8 | ACC:79.29+-1.06 | -nlayers_proj 1 -maskfeat_rate_1 0.5 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.5 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0005 -eval_freq 20 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256
# Run:8 | ACC:78.86+-1.51 | -nlayers_proj 1 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.1 -dropnode_rate_2 0.1 -lr 0.0005 -eval_freq 20 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256
# Run:8 | ACC:77.49+-0.73 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.4 -dropedge_rate_2 0.4 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256 -nlayers_enc 3
# Run:8 | ACC:78.44+-1.29 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256 -nlayers_enc 3
# Run:8 | ACC:79.04+-1.35 | -nlayers_proj 2 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256 -nlayers_enc 3
# Run:8 | ACC:79.62+-0.73 | -nlayers_proj 2 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.0 -emb_dim 256 -nlayers_enc 3
# Run:8 | ACC:78.85+-0.82 | -nlayers_proj 2 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 1.0 -beta 0.0 -gamma 0.1 -emb_dim 256 -nlayers_enc 3
# Run:8 | ACC:79.34+-0.86 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 1.0 -beta 0.1 -gamma 0.0 -emb_dim 256 -nlayers_enc 3
# Run:8 | ACC:78.53+-1.50 | -nlayers_proj 2 -maskfeat_rate_1 0.4 -maskfeat_rate_2 0.4 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 1.0 -beta 0.2 -gamma 0.0 -emb_dim 256 -nlayers_enc 3

# GREET: Run:10 | ACC:83.60+-0.50

###########################
# Trial:0 | ACC:79.30
# Trial:1 | ACC:79.50
# Trial:2 | ACC:80.60
# Trial:3 | ACC:81.20
# Trial:4 | ACC:78.40
# Trial:5 | ACC:80.70
# Trial:6 | ACC:78.80
# Trial:7 | ACC:79.70
# Trial:8 | ACC:79.40
# Trial:9 | ACC:78.60
# Trial:10 | ACC:79.00
# Trial:11 | ACC:81.00
# Trial:12 | ACC:79.50
# Trial:13 | ACC:77.80
# Trial:14 | ACC:80.60
# Trial:15 | ACC:78.00
# Trial:16 | ACC:80.90
# Trial:17 | ACC:78.40
# Trial:18 | ACC:78.20
# Trial:19 | ACC:80.10
# Trial:20 | ACC:77.20
# Trial:21 | ACC:79.70
# Trial:22 | ACC:79.50
# Trial:23 | ACC:80.50
# Trial:24 | ACC:79.50
# Trial:25 | ACC:80.30
# Trial:26 | ACC:81.10
# Trial:27 | ACC:78.80
# Trial:28 | ACC:78.70
# Trial:29 | ACC:78.00
# Trial:30 | ACC:79.60
# Trial:31 | ACC:80.40
# Trial:32 | ACC:80.10
# Trial:33 | ACC:81.60
# Trial:34 | ACC:78.80
# Trial:35 | ACC:80.90
# Trial:36 | ACC:77.90
# Trial:37 | ACC:78.60
# Trial:38 | ACC:79.10
# Trial:39 | ACC:78.90
# Trial:40 | ACC:79.30
# Trial:41 | ACC:78.30
# Trial:42 | ACC:80.90
# Trial:43 | ACC:79.00
# Trial:44 | ACC:79.60
# Trial:45 | ACC:80.60
# Trial:46 | ACC:79.70
# Trial:47 | ACC:77.50
# Trial:48 | ACC:80.50
# Trial:49 | ACC:80.30
# Trial:50 | ACC:80.50
# Trial:51 | ACC:81.80
# Trial:52 | ACC:80.20
# Trial:53 | ACC:80.30
# Trial:54 | ACC:77.60
# Trial:55 | ACC:81.30
# Trial:56 | ACC:78.40
# Trial:57 | ACC:81.70
# Trial:58 | ACC:80.50
# Trial:59 | ACC:75.20
# Trial:60 | ACC:79.80
# Trial:61 | ACC:79.80
# Trial:62 | ACC:79.10
# Trial:63 | ACC:77.00
# Trial:64 | ACC:78.50
# Trial:65 | ACC:76.50
# Trial:66 | ACC:79.70
# Trial:67 | ACC:78.60
# Trial:68 | ACC:78.60
# Trial:69 | ACC:80.20
# Trial:70 | ACC:80.10
# Trial:71 | ACC:80.40
# Trial:72 | ACC:80.60
# Trial:73 | ACC:79.50
# Trial:74 | ACC:79.20
# Trial:75 | ACC:80.50
# Trial:76 | ACC:79.40
# Trial:77 | ACC:78.80
# Trial:78 | ACC:77.70
# Trial:79 | ACC:80.60
# Trial:80 | ACC:80.10
# Trial:81 | ACC:78.90
# Trial:82 | ACC:79.90
# Trial:83 | ACC:78.30
# Trial:84 | ACC:81.10
# Trial:85 | ACC:79.90
# Trial:86 | ACC:79.00
# Trial:87 | ACC:81.80
# Trial:88 | ACC:78.90
# Trial:89 | ACC:78.20
# Trial:90 | ACC:79.90
# Trial:91 | ACC:79.70
# Trial:92 | ACC:78.30
# Trial:93 | ACC:80.30
# Trial:94 | ACC:79.00
# Trial:95 | ACC:79.80
# Trial:96 | ACC:80.00
# Trial:97 | ACC:81.30
# Trial:98 | ACC:78.60
# Trial:99 | ACC:77.00
# [FINAL RESULT] Dataset:cora | Run:100 | ACC:79.47+-1.23