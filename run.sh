# all the following commands are based on "DukeMTMC-reID". For "Market1501", please replace "dukemtmc" with "market1501".


# meta, with robust loss, with outlier
# Tab. 2, Row "No. 5" of our paper
python -W ignore meta_train_usl_knn_merge.py -d dukemtmc --logs-dir ./logs --data-dir ./data --symmetric

# meta, w/o robust loss, with outlier
# Tab. 2, Row "No. 4" of our paper
python -W ignore meta_train_usl_knn_merge.py -d dukemtmc --logs-dir ./logs --data-dir ./data

# w/o meta, with robust loss, with outlier
# Tab. 2, Row "No. 3" of our paper
python -W ignore train_usl_knn_merge.py -d dukemtmc --logs-dir ./logs --data-dir ./data --symmetric

# w/o meta, w/o robust loss, with outlier
# Tab. 2, Row "No. 2" of our paper
python -W ignore train_usl_knn_merge.py -d dukemtmc --logs-dir ./logs --data-dir ./data

# w/o meta, w/o outlier, w/o robust loss
# Tab. 2, Row "No. 1" of our paper
python -W ignore train_usl_no_outlier.py -d dukemtmc --logs-dir ./logs --data-dir ./data

# w/o meta, with outlier, with robust loss, with camera distribution align loss
# Tab. 1, Row "WFDR" of our paper
python -W ignore train_usl_knn_merge_cam.py -d dukemtmc --logs-dir ./logs --data-dir ./data --symmetric

# UDA, w/o meta, with outlier, with robust loss, w/o camera distribution align loss, MMT-pretrain
# Tab. 3, Row "MMT500+Ours" in our paper. The download link of "MMTD2M.pth" are provided in "Resource" of readme.md
python -W ignore meta_train_uda_knn_merge.py -s dukemtmc -t market1501 --logs-dir ./logs --data-dir ./data --symmetric --resume ./MMTD2M.pth
