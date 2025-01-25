# Cornell
python main.py -dataset cornell -ntrials 8 -sparse 0 -epochs 400 -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1

# Run:4 | ACC:63.51+-9.46 | -alpha 0.2 -beta 0.8 -gamma 0.0
# Run:4 | ACC:67.57+-8.76 | -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0004 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1 ***
# Run:8 | ACC:64.19+-5.53 | -alpha 0.1 -beta 0.7 -gamma 0.1
# Run:8 | ACC:63.51+-4.87 | -nlayers_enc 3 -emb_dim 256 -nlayers_enc 3 -alpha 0.1 -beta 0.8 -gamma 0.1
# Run:8 | ACC:66.22+-4.48 | -nlayers_enc 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.1 -dropnode_rate_2 0.1 -alpha 0.1 -beta 0.8 -gamma 0.1
# Run:8 | ACC:62.84+-7.49 | -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.1 -dropnode_rate_2 0.1 -lr 0.0001 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1
# Run:8 | ACC:66.22+-7.15 | -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1
# Run:8 | ACC:61.49+-5.01 | nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.0 -maskfeat_rate_2 0.0 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1
# Run:8 | ACC:62.16+-3.31 | -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1
# Run:8 | ACC:61.15+-5.56 | -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1
# Run:8 | ACC:62.16+-6.19 | -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1
# Run:8 | ACC:66.22+-7.15 | -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1
# Run:8 | ACC:62.16+-6.19 | -nlayers_enc 2 -emb_dim 256 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.1 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0001 -eval_freq 10 -alpha 0.1 -beta 0.8 -gamma 0.1

# Trial:0 | ACC:67.57
# Trial:1 | ACC:56.76
# Trial:2 | ACC:81.08 ***
# Trial:3 | ACC:64.86
# Trial:4 | ACC:62.16
# Trial:5 | ACC:62.16
# Trial:6 | ACC:72.97 ***
# Trial:7 | ACC:62.16
# Trial:8 | ACC:48.65
# Trial:9 | ACC:64.86
# Trial:10 | ACC:59.46
# Trial:11 | ACC:48.65
# Trial:12 | ACC:67.57
# Trial:13 | ACC:64.86
# Trial:14 | ACC:67.57
# Trial:15 | ACC:56.76
# Trial:16 | ACC:70.27 ***
# Trial:17 | ACC:51.35
# Trial:18 | ACC:56.76
# Trial:19 | ACC:56.76
# Trial:20 | ACC:56.76
# Trial:21 | ACC:54.05
# Trial:22 | ACC:64.86
# Trial:23 | ACC:64.86
# Trial:24 | ACC:64.86
# Trial:25 | ACC:62.16
# Trial:26 | ACC:62.16
# Trial:27 | ACC:62.16
# Trial:28 | ACC:54.05
# Trial:29 | ACC:64.86
# Trial:30 | ACC:72.97 ***
# Trial:31 | ACC:54.05
# Trial:32 | ACC:70.27 ***
# Trial:33 | ACC:56.76
# Trial:34 | ACC:64.86
# Trial:35 | ACC:67.57
# Trial:36 | ACC:59.46
# Trial:37 | ACC:59.46
# Trial:38 | ACC:56.76
# Trial:39 | ACC:56.76
# Trial:40 | ACC:70.27 ***
# Trial:41 | ACC:64.86
# Trial:42 | ACC:67.57
# Trial:43 | ACC:70.27 ***
# Trial:44 | ACC:67.57
# Trial:45 | ACC:54.05
# Trial:46 | ACC:62.16
# Trial:47 | ACC:59.46
# Trial:48 | ACC:56.76
# Trial:49 | ACC:62.16
# Trial:50 | ACC:59.46
# Trial:51 | ACC:45.95
# Trial:52 | ACC:62.16
# Trial:53 | ACC:64.86
# Trial:54 | ACC:62.16
# Trial:55 | ACC:64.86
# Trial:56 | ACC:54.05
# Trial:57 | ACC:59.46
# Trial:58 | ACC:51.35
# Trial:59 | ACC:59.46
# Trial:60 | ACC:62.16
# Trial:61 | ACC:62.16
# Trial:62 | ACC:64.86
# Trial:63 | ACC:67.57
# Trial:64 | ACC:64.86
# Trial:65 | ACC:70.27 ***
# Trial:66 | ACC:64.86
# Trial:67 | ACC:54.05
# Trial:68 | ACC:56.76
# Trial:69 | ACC:67.57
# Trial:70 | ACC:59.46
# Trial:71 | ACC:64.86
# Trial:72 | ACC:70.27 ***
# Trial:73 | ACC:64.86
# Trial:74 | ACC:67.57
# Trial:75 | ACC:64.86
# Trial:76 | ACC:56.76
# Trial:77 | ACC:62.16
# Trial:78 | ACC:62.16
# Trial:79 | ACC:56.76
# Trial:80 | ACC:51.35
# Trial:81 | ACC:62.16
# Trial:82 | ACC:72.97 ***
# Trial:83 | ACC:56.76
# Trial:84 | ACC:59.46
# Trial:85 | ACC:62.16
# Trial:86 | ACC:59.46
# Trial:87 | ACC:59.46
# Trial:88 | ACC:59.46
# Trial:89 | ACC:59.46
# Trial:90 | ACC:70.27 ***
# Trial:91 | ACC:56.76
# Trial:92 | ACC:64.86
# Trial:93 | ACC:62.16
# Trial:94 | ACC:67.57
# Trial:95 | ACC:64.86
# Trial:96 | ACC:70.27 ***
# Trial:97 | ACC:59.46
# Trial:98 | ACC:51.35
# Trial:99 | ACC:62.16


# Run:8 | ACC:63.85+-5.39 | -alpha 0.0 -beta 0.8 -gamma 0.2
# Trial:0 | ACC:70.27
# Trial:1 | ACC:56.76
# Trial:2 | ACC:62.16
# Trial:3 | ACC:70.27
# Trial:4 | ACC:67.57
# Trial:5 | ACC:56.76
# Trial:6 | ACC:67.57
# Trial:7 | ACC:59.46
