## Environment

Create a virtual environment with pytorch, pytorch-lightning, mne-python (basic functionnalities) and colorednoise.

```
conda env create -n pt_env
conda activate pt_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mne
pip install pytorch-lightning
pip install colorednoise
pip install tensorboard
```

## Training

Three models available: 1dcnn, lstm and deepsif.

- In the main_train.py : change the path to the data repository (path for SEREEGA data, NMM data and results)
- Then run the code with the proper parameters
  Example for the 1dcnn, on the _dps_sereega_2src_om3_train_big_ dataset, with 100 samples :

```
python main_train.py _dps_sereega_2src_om3_train_big_ sereega -source_space fsav_994 -spikes_folder nmm_spikes_nov23_train -model 1dcnn -to_load 100 -eeg_snr 5 -inter_layer 2048 -kernel_size 5 -batch_size 8 -n_epochs 100 -scaler linear -loss cosine -sfolder train_size_impact
```

## Eval

Example of command for evaluation :

```
python eval.py _dps_sereega_2src_om3_test_ sereega -orientation constrained -electrode_montage standard_1020 -source_space fsav_994 -spikes_folder nmm_spikes_nov23_test -eeg_snr 5 -to_load 2000 -per_valid 1 -sfolder train_size_impact_res -train_simu_name _dps_sereega_2src_om3_train_big_ -train_simu_type sereega -n_train_samples 3200 -train_sfolder train_size_impact

python inverse_problem/main_train.py mes_debug_python -root_simu D:/fyp/stESI_pub -results_path D:/fyp/stESI_pub/inverse_problem/results -simu_type sereega  -source_space fsav_994 -electrode_montage standard_1020 -orientation constrained -spikes_folder nmm_spikes_nov23 -model DEEPSIF -to_load 100 -per_valid 0.2 -n_times 500 -eeg_snr 5 -loss cosine -scaler linear -deepsif_temporal_input_size 500
```

python eval.py mes*debug_python -eval_simu_type sereega -root_simu "C:/Users/pasin/Documents/Data/simulation/fsaverage" -results_path "D:/fyp/stESI_pub/inverse_problem/results" -orientation constrained -electrode_montage standard_1020 -source_space fsav_994 -eeg_snr 5 -to_load 100 -per_valid 1 -mets deepsif -train_run_dir "D:\fyp\stESI_pub\inverse_problem\results\mes_debug_pythonfsav_994*\trainings\simu_sereega_srcspace_fsav_994_model_DEEPSIF_trainset_80_epochs_5_loss_cosine_norm_linear"

python eval*real.py mes_debug_python -root_simu "C:/Users/pasin/Documents/Data/simulation/fsaverage" -subject_name fsaverage -orientation constrained -electrode_montage standard_1020 -source_space fsav_994 -n_times 500 -to_load 25 -leadfield_mat "D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat" -train_run_dir "D:\fyp\stESI_pub\inverse_problem\results\mes_debug_pythonfsav_994*\trainings\simu_sereega_srcspace_fsav_994_model_DEEPSIF_trainset_80_epochs_100_loss_cosine_norm_linear" -out_mat "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994/eval_real_all_out_deepSIF_check_20.mat"

python eval.py mes*debug_python -eval_simu_type SEREEGA -root_simu "C:/Users/pasin/Documents/Data/simulation/fsaverage" -results_path "D:/fyp/stESI_pub/inverse_problem/results" -subject_name fsaverage -orientation constrained -electrode_montage standard_1020 -source_space fsav_994 -eeg_snr 5 -to_load 100 -per_valid 1 -mets deepsif -leadfield_mat "D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat" -train_run_dir "D:\fyp\stESI_pub\inverse_problem\results\mes_debug_pythonfsav_994*\trainings\simu_sereega_srcspace_fsav_994_model_DEEPSIF_trainset_80_epochs_5_loss_cosine_norm_linear"

python eval*real_vit.py mes_debug_python -root_simu "C:/Users/pasin/Documents/Data/simulation/fsaverage" -subject_name fsaverage -orientation constrained -electrode_montage standard_1020 -source_space fsav_994 -n_times 500 -to_load 10 -leadfield_mat "D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat" -train_run_dir "D:\fyp\stESI_pub\inverse_problem\results\mes_debug_pythonfsav_994*\trainings\simu*sereega_srcspace_fsav_994_model_VIT_trainset_80_epochs_100_loss_cosine_norm_linear" -out_mat "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/eval_real_all_out_vit_new.mat"

python main_train.py mes_debug_python -root_simu D:/... -results_path D:/... -simu_type sereega \
 -source_space fsav_994 -electrode_montage standard_1020 -orientation constrained \
 -model STCT -n_times 500 -to_load 100 -per_valid 0.2 -eeg_snr 5 -loss cosine -scaler linear \
 -st_embed_dim 256 -st_depth 6 -st_heads 8 -st_mlp_dim 512 -st_dropout 0.1 \
 -st_spatial_hidden 64 -st_spatial_kernel 5

python inverse_problem/main_train.py mes_debug_python -root_simu D:/fyp/stESI_pub -results_path D:/fyp/stESI_pub/inverse_problem/results -simu_type sereega -source_space fsav_994 -electrode_montage standard_1020 -orientation constrained -spikes_folder nmm_spikes_nov23 -model STCT -to_load 100 -per_valid 0.2 -n_times 500 -eeg_snr 5 -loss cosine -scaler linear -st_embed_dim 256 -st_depth 6 -st_heads 8 -st_mlp_dim 512 -st_dropout 0.1 -st_spatial_hidden 64 -st_spatial_kernel 5

python inverse_problem/main_train.py mes_debug_python -root_simu D:/fyp/stESI_pub -results_path D:/fyp/stESI_pub/inverse_problem/results -simu_type sereega -source_space fsav_994 -electrode_montage standard_1020 -orientation constrained -spikes_folder nmm_spikes_nov23 -model STCT -to_load 100 -per_valid 0.2 -n_times 500 -eeg_snr 5 -loss cosine -scaler linear -leadfield_mat D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat -st_embed_dim 256 -st_depth 6 -st_heads 8 -st_mlp_dim 512 -st_dropout 0.1 -st_spatial_hidden 64 -st_spatial_kernel 5

python eval.py mes*debug_python -eval_simu_type SEREEGA -root_simu "C:/Users/pasin/Documents/Data/simulation/fsaverage" -results_path "D:/fyp/stESI_pub/inverse_problem/results" -subject_name fsaverage -orientation constrained -electrode_montage standard_1020 -source_space fsav_994 -eeg_snr 5 -to_load 100 -per_valid 1 -mets stct -leadfield_mat "D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat" -train_run_dir "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simu_sereega_srcspace_fsav_994_model_STCT_trainset_80_epochs_25_loss_cosine_norm_linear"

# run with check points

python eval*real_stct.py mes_debug_python -real_data_dir "D:/fyp/stESI_pub/real_data" -real_data_glob "eeg_and_src_data*\*.mat" -n*times 500 -to_load 10 -leadfield_mat "D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat" -train_run_dir "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simu*sereega_srcspace_fsav_994_model_STCT_trainset_80_epochs_25_loss_cosine_norm_linear" -ckpt_path "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simu*sereega_srcspace_fsav_994_model_STCT_trainset_80_epochs_25_loss_cosine_norm_linear/pl_checkpoints/<YOUR_CKPT>.ckpt" -out_mat "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/eval_real_all_out_stct_new.mat"

python eval_real_vit.py mes_debug_python -real_data_dir "D:/fyp/stESI_pub/real_data" -real_data_glob "eeg_and_src_data\*.mat" -ntimes 500 -to_load 10 -leadfield_mat "D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat" -train_run_dir "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simu*sereega_srcspace_fsav_994_model_VIT_trainset_80000_epochs_100_loss_cosine_norm_linear" -ckpt_path "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simu\sereega*srcspace_fsav_994_model_VIT_trainset_80000_epochs_100_loss_cosine_norm_linear/pl_checkpoints/pl_checkpoints_epoch=38-train_loss=-0.12.ckpt" -out_mat "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994\*/eval_real_all_out_vit_sample0.mat"

python eval*real_vit.py mes_debug_python -real_data_dir "D:/fyp/stESI_pub/real_data" -real_data_glob "eeg_and_src_data\*.mat" -n_times 500 -to_load 10 -leadfield_mat "D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat" -train_run_dir "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simusereega_srcspace_fsav_994_model_VIT_trainset_80000_epochs_100_loss_cosine_norm_linear" -ckpt_path "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994\*/trainings/simu_sereega_srcspace_fsav_994_model_VIT_trainset_80000_epochs_100_loss_cosine_norm_linear/pl_checkpoints/pl_checkpoints_epoch=38-train_loss=-0.12.ckpt" -out_mat "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994/eval_real_all_out_vit_80000_38.mat"

python eval*real_vit.py mes_debug_python -real_data_dir "D:/fyp/stESI_pub/real_data" -real_data_glob "eeg_and_src_data*\*.mat" -n*times 500 -to_load 10 -leadfield_mat "D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat" -train_run_dir "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simu*sereega_srcspace_fsav_994_model_VIT_trainset_80000_epochs_100_loss_cosine_norm_linear" -ckpt_path "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simu*sereega_srcspace_fsav_994_model_VIT_trainset_80000_epochs_100_loss_cosine_norm_linear/pl_checkpoints/epoch=38-train_loss=-0.12.ckpt" -out_mat "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/eval_real_all_out_vit_80000_38.ma

D:\fyp\stESI*pub\inverse_problem>python eval_real_vit.py mes_debug_python -real_data_dir "D:/fyp/stESI_pub/real_data" -real_data_glob "eeg_and_src_data*\*.mat" -n*times 500 -to_load 10 -leadfield_mat "D:/fyp/stESI_pub/anatomy/leadfield_75_20k.mat" -train_run_dir "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simu*sereega*srcspace_fsav_994_model_VIT_trainset_80000_epochs_100_loss_cosine_norm_linear" -ckpt_path "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994*/trainings/simu*sereega_srcspace_fsav_994_model_VIT_trainset_80000_epochs_100_loss_cosine_norm_linear/pl_checkpoints/epoch=38-train_loss=-0.12.ckpt" -out_mat "D:/fyp/stESI_pub/inverse_problem/results/mes_debug_pythonfsav_994\*/eval_real_all_out_vit_80000_38.mat"
