# Texas
python main_log.py -dataset texas -ntrials 8 -sparse 0 -epochs 400 -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 10 -alpha 0.5 -beta 0.5 -gamma 0.5 -emb_dim 256

# ACC:85.14+-1.35 | -alpha 0.5 -beta 0.5 -gamma 0.5
# Trial:0 | ACC:83.78
# Trial:1 | ACC:86.49
# Trial:2 | ACC:62.16
# Trial:3 | ACC:86.49
# Trial:4 | ACC:86.49
# Trial:5 | ACC:78.38
# Trial:6 | ACC:75.68
# Trial:7 | ACC:64.86

# Run:8 | ACC:77.03+-6.19 | -alpha 0.5 -beta 0.5 -gamma 0.4
# Run:8 | ACC:77.03+-6.76 | -alpha 0.6 -beta 0.5 -gamma 0.5
# Run:8 | ACC:75.68+-9.4 (trial: 89.19) | -alpha 0.6 -beta 0.5 -gamma 0.4
# Run:8 | ACC:75.00+-7.25 | -alpha 0.6 -beta 0.6 -gamma 0.4
# Run:8 | ACC:75.68+-6.34 | -alpha 0.5 -beta 0.5 -gamma 0.2
# Run:8 | ACC:78.04+-9.21 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -alpha 0.5 -beta 0.5 -gamma 0.5 ***
# Run:8 | ACC:73.31+-6.82 | -nlayers_proj 2 -maskfeat_rate_1 0.0 -maskfeat_rate_2 0.0 -dropedge_rate_1 0.2 -dropedge_rate_2 0.2 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5
# Run:8 | ACC:73.99+-6.47 | nlayers_proj 2 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5
# Run:8 | ACC:75.00+-4.83 | -nlayers_proj 2 -maskfeat_rate_1 0.0 -maskfeat_rate_2 0.0 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5
# Run:8 | ACC:77.36+-8.32 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.0 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5
#  Run:8 | ACC:77.36+-8.32 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.0 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5
# Run:8 | ACC:75.00+-4.83 | -nlayers_proj 2 -maskfeat_rate_1 0.05 -maskfeat_rate_2 0.05 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5 
# Run:20 | ACC:79.19+-6.46 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5 -emb_dim 256
# Run:20 | ACC:78.11+-6.56 | -nlayers_proj 2 -maskfeat_rate_1 0.2 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5 -emb_dim 256
# Run:20 | ACC:79.19+-5.06 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.0005 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5 -emb_dim 512
# Run:20 | ACC:79.86+-6.65 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5 -emb_dim 256 ***
# Trial:0 | ACC:86.49
# Trial:1 | ACC:83.78
# Trial:2 | ACC:70.27
# Trial:3 | ACC:89.19
# Trial:4 | ACC:89.19
# Trial:5 | ACC:81.08
# Trial:6 | ACC:70.27
# Trial:7 | ACC:72.97
# Trial:8 | ACC:83.78
# Trial:9 | ACC:81.08
# Trial:10 | ACC:83.78
# Trial:11 | ACC:83.78
# Trial:12 | ACC:75.68
# Trial:13 | ACC:83.78
# Trial:14 | ACC:91.89
# Trial:15 | ACC:72.97
# Trial:16 | ACC:72.97
# Trial:17 | ACC:78.38
# Trial:18 | ACC:72.97
# Trial:19 | ACC:72.97

# Run:8 | ACC:79.39+-6.03 | -nlayers_proj 2 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.2 -dropedge_rate_1 0.0 -dropedge_rate_2 0.0 -dropnode_rate_1 0.0 -dropnode_rate_2 0.0 -lr 0.001 -eval_freq 20 -alpha 0.5 -beta 0.5 -gamma 0.5
# Trial:0 | ACC:83.78
# Trial:1 | ACC:81.08
# Trial:2 | ACC:78.38
# Trial:3 | ACC:78.38
# Trial:4 | ACC:91.89 *
# Trial:5 | ACC:75.68
# Trial:6 | ACC:75.68
# Trial:7 | ACC:70.27
# Trial:8 | ACC:75.68
# Trial:9 | ACC:81.08
# Trial:10 | ACC:86.49 *
# Trial:11 | ACC:78.38
# Trial:12 | ACC:67.57
# Trial:13 | ACC:81.08
# Trial:14 | ACC:81.08
# Trial:15 | ACC:70.27
# Trial:16 | ACC:72.97
# Trial:17 | ACC:75.68
# Trial:18 | ACC:78.38
# Trial:19 | ACC:83.78
# Trial:20 | ACC:89.19 *
# Trial:21 | ACC:81.08
# Trial:22 | ACC:64.86
# Trial:23 | ACC:83.78
# Trial:24 | ACC:78.38
# Trial:25 | ACC:75.68
# Trial:26 | ACC:67.57
# Trial:27 | ACC:72.97
# Trial:28 | ACC:75.68
# Trial:29 | ACC:75.68
# Trial:30 | ACC:83.78
# Trial:31 | ACC:83.78
# Trial:32 | ACC:81.08
# Trial:33 | ACC:78.38
# Trial:34 | ACC:83.78
# Trial:35 | ACC:81.08
# Trial:36 | ACC:70.27
# Trial:37 | ACC:72.97
# Trial:38 | ACC:75.68
# Trial:39 | ACC:70.27
# Trial:40 | ACC:86.49 *
# Trial:41 | ACC:83.78
# Trial:42 | ACC:70.27
# Trial:43 | ACC:75.68
# Trial:44 | ACC:83.78
# Trial:45 | ACC:70.27
# Trial:46 | ACC:72.97
# Trial:47 | ACC:78.38
# Trial:48 | ACC:81.08
# Trial:49 | ACC:75.68
# Trial:50 | ACC:86.49 *
# Trial:51 | ACC:75.68
# Trial:52 | ACC:75.68
# Trial:53 | ACC:81.08
# Trial:54 | ACC:67.57
# Trial:55 | ACC:75.68
# Trial:56 | ACC:75.68
# Trial:57 | ACC:78.38
# Trial:58 | ACC:70.27
# Trial:59 | ACC:72.97
# Trial:60 | ACC:83.78
# Trial:61 | ACC:83.78
# Trial:62 | ACC:67.57
# Trial:63 | ACC:83.78
# Trial:64 | ACC:81.08
# Trial:65 | ACC:67.57
# Trial:66 | ACC:70.27
# Trial:67 | ACC:75.68
# Trial:68 | ACC:81.08
# Trial:69 | ACC:75.68
# Trial:70 | ACC:83.78
# Trial:71 | ACC:75.68
# Trial:72 | ACC:72.97
# Trial:73 | ACC:83.78
# Trial:74 | ACC:86.49 *
# Trial:75 | ACC:75.68
# Trial:76 | ACC:67.57
# Trial:77 | ACC:62.16
# Trial:78 | ACC:75.68
# Trial:79 | ACC:78.38
# Trial:80 | ACC:81.08
# Trial:81 | ACC:78.38
# Trial:82 | ACC:70.27
# Trial:83 | ACC:86.49 *
# Trial:84 | ACC:83.78
# Trial:85 | ACC:75.68
# Trial:86 | ACC:72.97
# Trial:87 | ACC:72.97
# Trial:88 | ACC:81.08
# Trial:89 | ACC:75.68
# Trial:90 | ACC:83.78
# Trial:91 | ACC:83.78
# Trial:92 | ACC:64.86
# Trial:93 | ACC:86.49 *
# Trial:94 | ACC:72.97
# Trial:95 | ACC:75.68
# Trial:96 | ACC:83.78
# Trial:97 | ACC:72.97
# Trial:98 | ACC:78.38
# Trial:99 | ACC:78.38
# Trial:100 | ACC:86.49 *
# Trial:101 | ACC:75.68
# Trial:102 | ACC:75.68
# Trial:103 | ACC:72.97
# Trial:104 | ACC:72.97
# Trial:105 | ACC:70.27
# Trial:106 | ACC:72.97
# Trial:107 | ACC:70.27
# Trial:108 | ACC:83.78
# Trial:109 | ACC:78.38
# Trial:110 | ACC:83.78
# Trial:111 | ACC:78.38
# Trial:112 | ACC:67.57
# Trial:113 | ACC:78.38
# Trial:114 | ACC:78.38
# Trial:115 | ACC:78.38
# Trial:116 | ACC:70.27
# Trial:117 | ACC:70.27
# Trial:118 | ACC:81.08
# Trial:119 | ACC:81.08
# Trial:120 | ACC:89.19 *
# Trial:121 | ACC:81.08
# Trial:122 | ACC:64.86
# Trial:123 | ACC:83.78
# Trial:124 | ACC:83.78
# Trial:125 | ACC:75.68
# Trial:126 | ACC:70.27
# Trial:127 | ACC:78.38
# Trial:128 | ACC:75.68
# Trial:129 | ACC:72.97
# Trial:130 | ACC:83.78
# Trial:131 | ACC:78.38
# Trial:132 | ACC:70.27
# Trial:133 | ACC:83.78
# Trial:134 | ACC:75.68
# Trial:135 | ACC:78.38
# Trial:136 | ACC:78.38
# Trial:137 | ACC:72.97
# Trial:138 | ACC:75.68
# Trial:139 | ACC:75.68
# Trial:140 | ACC:78.38
# Trial:141 | ACC:83.78
# Trial:142 | ACC:72.97
# Trial:143 | ACC:81.08
# Trial:144 | ACC:83.78
# Trial:145 | ACC:72.97
# Trial:146 | ACC:75.68
# Trial:147 | ACC:67.57
# Trial:148 | ACC:81.08
# Trial:149 | ACC:72.97
# Trial:150 | ACC:86.49 *
# Trial:151 | ACC:81.08
# Trial:152 | ACC:72.97
# Trial:153 | ACC:83.78
# Trial:154 | ACC:75.68
# Trial:155 | ACC:81.08
# Trial:156 | ACC:70.27
# Trial:157 | ACC:75.68
# Trial:158 | ACC:75.68
# Trial:159 | ACC:72.97
# Trial:160 | ACC:86.49 *
# Trial:161 | ACC:83.78
# Trial:162 | ACC:67.57
# Trial:163 | ACC:78.38
# Trial:164 | ACC:75.68
# Trial:165 | ACC:75.68
# Trial:166 | ACC:75.68
# Trial:167 | ACC:62.16
# Trial:168 | ACC:81.08
# Trial:169 | ACC:72.97
# Trial:170 | ACC:78.38
# Trial:171 | ACC:81.08
# Trial:172 | ACC:75.68
# Trial:173 | ACC:81.08
# Trial:174 | ACC:78.38
# Trial:175 | ACC:70.27
# Trial:176 | ACC:72.97
# Trial:177 | ACC:64.86
# Trial:178 | ACC:70.27
# Trial:179 | ACC:72.97
# Trial:180 | ACC:86.49 *
# Trial:181 | ACC:83.78
# Trial:182 | ACC:67.57
# Trial:183 | ACC:78.38
# Trial:184 | ACC:75.68
# Trial:185 | ACC:75.68
# Trial:186 | ACC:75.68
# Trial:187 | ACC:70.27
# Trial:188 | ACC:81.08
# Trial:189 | ACC:78.38
# Trial:190 | ACC:83.78
# Trial:191 | ACC:83.78
# Trial:192 | ACC:64.86
# Trial:193 | ACC:70.27
# Trial:194 | ACC:86.49 *
# Trial:195 | ACC:75.68
# Trial:196 | ACC:72.97
# Trial:197 | ACC:70.27
# Trial:198 | ACC:83.78
# Trial:199 | ACC:81.08

