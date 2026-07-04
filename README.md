# Transformer-based Reconstruction of Canopy Profiles from Large-Footprint Waveform LiDAR
This repository contains the codebase for training and applying the encoder–decoder Transformer framework developed by Siddiqui et al. (2026) to reconstruct high-resolution canopy profiles from large-footprint waveform LiDAR. The model uses co-located, high-density airborne LiDAR profiles as reference data to recover fine-scale vertical canopy structure from high-altitude waveform observations.

For details on the study objectives, model architecture, training procedure, and evaluation framework, please refer to the preprint: https://doi.org/10.31223/X5SJ6B.

<img width="2360" height="1156" alt="Fig4_LVIS2ALS_architecture" src="https://github.com/user-attachments/assets/f0246083-90c9-4132-bc7b-7beaaa40e441" />
<i>Figure 1. Transformer-based canopy profile reconstruction architecture. Example input LVIS and ground-truth ALS waveforms are shown from a real paired observation. The predicted ALS waveform is illustrated by recoloring the ground truth for visualization purposes.</i>
