CUDA_VISIBLE_DEVICES=2 python train_validation_model.py  --strategy five_fold --batch_size 64  --overlap with --save_file 1123_transformer_WOGCN_leftright_time_channel_paraUpdate_bs64 --log_dir 1114_five_fold --lamda 0 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
CUDA_VISIBLE_DEVICES=2 python train_validation_model.py  --strategy five_fold --batch_size 64  --overlap with --save_file 1123_transformer_WOGCN_leftright_time_channel_paraUpdate_bs64 --log_dir 1114_five_fold --lamda 0 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
CUDA_VISIBLE_DEVICES=2 python train_validation_model.py  --strategy five_fold --batch_size 64  --overlap with --save_file 1123_transformer_WOGCN_leftright_time_channel_paraUpdate_bs64 --log_dir 1114_five_fold --lamda 0 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
CUDA_VISIBLE_DEVICES=2 python train_validation_model.py  --strategy five_fold --batch_size 64  --overlap with --save_file 1123_transformer_WOGCN_leftright_time_channel_paraUpdate_bs64 --log_dir 1114_five_fold --lamda 0 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
CUDA_VISIBLE_DEVICES=2 python train_validation_model.py  --strategy five_fold --batch_size 64  --overlap with --save_file 1123_transformer_WOGCN_leftright_time_channel_paraUpdate_bs64 --log_dir 1114_five_fold --lamda 0 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
