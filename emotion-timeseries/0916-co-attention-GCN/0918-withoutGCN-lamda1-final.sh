#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.0 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.0 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.0 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.0 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.0 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat


#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.1 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.2 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.3 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.4 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.5 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.6 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.7 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.8 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.9 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 64 --LSTM_hidden 1024 --overlap with --save_file 0918_final_result_lamda1_win120_over30 --log_dir 0916_five_fold --lamda 1.0 --fold_id 1 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat



#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.1 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.2 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.3 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.4 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.5 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.6 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.7 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.8 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.9 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 64 --LSTM_hidden 1024 --overlap with --save_file 0918_final_result_lamda1_win120_over30 --log_dir 0916_five_fold --lamda 1.0 --fold_id 2 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat



#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.1 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.2 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.3 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.4 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.5 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.6 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.7 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.8 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.9 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 64 --LSTM_hidden 1024 --overlap with --save_file 0918_final_result_lamda1_win120_over30 --log_dir 0916_five_fold --lamda 1.0 --fold_id 3 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat



#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.1 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.2 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.3 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.4 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.5 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.6 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.7 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.8 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.9 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 64 --LSTM_hidden 1024 --overlap with --save_file 0918_final_result_lamda1_win120_over30 --log_dir 0916_five_fold --lamda 1.0 --fold_id 4 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat



#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.1 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.2 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.3 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.4 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.5 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.6 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat

#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.7 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.8 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
#CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 32 --overlap without --save_file 0409_win20_WOO_multilabelMetrics_FF_withdecoder_withoutGCN --log_dir 0406_five_fold --lamda 0.9 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
#
CUDA_VISIBLE_DEVICES=1 python train_co_attn.py  --strategy five_fold --batch_size 64 --LSTM_hidden 1024 --overlap with --save_file 0918_final_result_lamda1_win120_over30 --log_dir 0916_five_fold --lamda 1.0 --fold_id 5 --path1 ../EEG_PSD_9_DOM/   --path2 ../DOM_feature_all/DOM_featureAll.mat
