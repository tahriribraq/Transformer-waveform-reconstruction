#!/bin/bash -l

spack env activate autoencoder-x86_64-25020401
time python scripts/train_best_config.py --data_dir ./data --lvis_file LVIS-F_preprocessed_waveforms.pkl --als_file AOP_ALS_canopy_profiles.pkl --checkpoint_dir ./checkpoints_best_config > best_config.log

time python scripts/evaluate.py --data_dir ./data --lvis_file LVIS-F_preprocessed_waveforms.pkl --als_file AOP_ALS_canopy_profiles.pkl --checkpoint ./checkpoints_best_config/best.pt >> best_config.log
