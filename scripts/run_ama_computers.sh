# Amazon Computers
python main.py -dataset computers -ntrials 10 -sparse 1 -epochs 1500 -cl_batch_size 5000 -nlayers_proj 1 -alpha 0.3 -k 10 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.5 -dropedge_rate_2 0.1 -lr_disc 0.0001 -margin_hom 0.1 -margin_het 0.5 -cl_rounds 3 -eval_freq 20