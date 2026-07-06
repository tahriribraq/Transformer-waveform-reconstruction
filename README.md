# Transformer-based Reconstruction of Canopy Profiles from Large-Footprint Waveform LiDAR
This repository contains the codebase for training and applying the encoder–decoder Transformer framework developed by Siddiqui et al. (2026) to reconstruct high-resolution canopy profiles from large-footprint waveform LiDAR. The model uses co-located, high-density airborne LiDAR profiles as reference data to recover fine-scale vertical canopy structure from high-altitude waveform observations. All code was implemented in Python 3.13.3.

For details on the study objectives, model architecture, training procedure, and evaluation framework, please refer to the preprint: https://doi.org/10.31223/X5SJ6B.

<img width="2360" height="1156" alt="Fig4_LVIS2ALS_architecture" src="https://github.com/user-attachments/assets/f0246083-90c9-4132-bc7b-7beaaa40e441" />
<i>Figure. Transformer-based canopy profile reconstruction architecture. The example LVIS input and ALS reference profiles are from a real paired observation. The predicted ALS profile is illustrated by recoloring the ground truth for visualization purposes.</i>

<hr>

# Data preprocessing script
[denoise_clip_waveforms.ipynb](waveform%20preprocessing/denoise_clip_waveforms.ipynb): This notebook implements the LVIS waveform preprocessing workflow used to smooth, threshold, and clip raw waveforms in order to suppress background noise and isolate the signal portion of each waveform. The workflow assumes that LVIS waveform data have been loaded into pandas DataFrames. Raw waveform information from the LVIS L1B [1] and L2 [2] data products was downloaded from the National Snow and Ice Data Center, loaded into DataFrames, and subset to retain shots within the Smithsonian Environmental Research Center (SERC) site boundary using polygons provided through the NEON data portal [3]. The resulting site-level waveform data was then processed using the methods provided in this notebook.

<hr>

# Dataset loading, model architecture, training, and evaluation files
- [dataset.py](waveform%20preprocessing/dataset.py): Defines the dataset-loading utilities used to prepare paired LVIS–ALS samples for model training and evaluation. This script maps ALS profiles to the fixed output height grid, creates masks for valid LVIS waveform regions, handles variable-length LVIS input sequences through padding, and partitions the dataset into training, validation, and test subsets.

- **model folder:** Contains the PyTorch modules that define the encoder–decoder Transformer architecture. The complete model is assembled in [transformer.py](model/transformer.py).

- **training folder:** Contains the main training utilities. [config.py](training/config.py) defines configurable hyperparameters; [losses.py](training/losses.py) implements the training loss functions; [metrics.py](training/metrics.py) provides evaluation metrics; and [trainer.py](training/trainer.py) contains the training, validation, testing, and checkpoint-saving routines.

- [evaluate.py](scripts/evaluate.py): Provides methods for loading trained model checkpoints, generating predictions, and computing evaluation metrics on the held-out test set.

- **best-performing model:** [train_best_config.py](scripts/train_best_config.py) trains the model using the best-performing hyperparameter configuration reported in the manuscript. The corresponding hyperparameters are defined in the 'TrainingConfig' object within the main function. [submit_best_config.sh](slurm%20jobs/submit_best_config.sh) is the SLURM submission script used to run this training job on the RIT research computing cluster [4]. [best.pt](checkpoints_best_config/best.pt) contains the checkpoint for the best-performing model, and [best_config.log](logs/best_config.log) contains the training and evaluation log generated from this run.

<hr>

# Comparison with waveform deconvolution algorithms
[Deconv_file.ipynb](waveform%20deconvolution/Deconv_file.ipynb) implements the Gold and Richardson–Lucy waveform deconvolution algorithms, adapted from Bhatta et al. [5], and applies them to the preprocessed LVIS waveforms. The notebook compares the deconvolved waveforms with both the ALS reference profiles and the Transformer-reconstructed profiles by computing normalized profile correlations.

The notebook also evaluates canopy structural complexity (CSC) metric recovery by deriving FHD and VCR from the deconvolved waveforms and comparing them with the corresponding metrics derived from the ALS reference profiles and Transformer-reconstructed profiles.

<hr>

# References
1. Blair, J. B. & Hofton, M. (2020). LVIS Facility L1B Geolocated Return Energy Waveforms. (LVISF1B, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/XQJ8PN8FTIDG. Date Accessed 07-05-2026.
2. Blair, J. B. & Hofton, M. (2020). LVIS Facility L2 Geolocated Surface Elevation and Canopy Height Product. (LVISF2, Version 1). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/VP7J20HJQISD. Date Accessed 07-05-2026.
3. All_NEON_TOS_Plots_V10. Available online: https://data.neonscience.org/documents/-/document_library_display/kV4WWrbEEM2s/view_file/3411434. Date Accessed 07-05-2026.
4. Rochester Institute of Technology. Research Computing Services. Rochester Institute of Technology. https://doi.org/10.34788/0S3G-QD15.
5. Bhatta, R. (2026). Waveform-LiDAR-Analysis: Novel-LAI-estimation (Version main) [Computer software]. GitHub. https://github.com/Bhatta6190/Waveform-LiDAR-Analysis.
