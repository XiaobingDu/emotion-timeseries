CUDA_VISIBLE_DEVICES=1 python train_validation_model.py  --strategy five_fold --batch_size 64  --overlap with --save_file 1201_transformer_timestep_channel_labelCorr_All_mask --log_dir 1114_five_fold --lamda 0 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
CUDA_VISIBLE_DEVICES=1 python train_validation_model.py  --strategy five_fold --batch_size 64  --overlap with --save_file 1201_transformer_timestep_channel_labelCorr_All_mask --log_dir 1114_five_fold --lamda 0 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
CUDA_VISIBLE_DEVICES=1 python train_validation_model.py  --strategy five_fold --batch_size 64  --overlap with --save_file 1201_transformer_timestep_channel_labelCorr_All_mask --log_dir 1114_five_fold --lamda 0 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
CUDA_VISIBLE_DEVICES=1 python train_validation_model.py  --strategy five_fold --batch_size 64  --overlap with --save_file 1201_transformer_timestep_channel_labelCorr_All_mask --log_dir 1114_five_fold --lamda 0 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
