# PubMed
python main.py -dataset pubmed -ntrials 10 -sparse 1 -epochs 800 -cl_batch_size 5000 -nlayers_proj 2 -alpha 0.1 -k 0 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.1 -lr_disc 0.001 -margin_hom 0.5 -margin_het 0.5 -cl_rounds 2  -eval_freq 20