
# DeepENF: A data-driven Electric Network Frequency estimation framework

This repository contains the code for the paper *DeepENF: A data-driven Electric Network Frequency estimation framework* introduced
by Ioannis Tsingalis and Constantine Kotropoulos

## Abstract

This paper proposes DeepENF, an effective Convolutional Neural Network (CNN) framework, for estimating the Electric Network Frequency (ENF). ENF has gained significant importance in forensics, playing a crucial role in various applications such as device identification, verifying the recording time, estimating recording locations, etc.
DeepENF achieves state-of-the-art ENF estimation accuracy using a CNN that relies on a single ENF harmonic. This single-harmonic approach simplifies the architecture and reduces the computational cost of DeepENF compared to other ENF estimation techniques with similar performance, which often require multiple ENF harmonics. Additionally, using a single ENF harmonic eliminates the need for fine-tuning to determine the number and combination of harmonics for ENF estimation. These advantages make DeepENF particularly appealing to practitioners. Consequently, DeepENF offers a more efficient and accessible solution for accurate ENF estimation in practical applications.
DeepENF is evaluated using benchmark audio recordings from the ENF-WHU dataset, highlighting its proficiency in extracting the ENF signal from possibly noisy observations. This is the first study to leverage CNN for ENF estimation utilizing raw input signals.

## Usage
### 1. Repository preparation 
Clone the repository, extract the files from [here](https://drive.google.com/file/d/1l1AF3JNrQE-7KJXyE9YYLcZr86DX8M5g/view?usp=sharing) in the folder `/DeepENF/experiments/`. The folders should have the structure

```angular2html
├── Code
│   └── .py files
├── Data
│   ├── Folds
│   └── Hua
├── experiments
│   ├── RFA
```

### 2. Install requirements
The requirements are in the *requirements.txt* file.

### 3. Download datasets
You can download the ENF-WHU-Dataset from [here](https://github.com/ghua-ac/ENF-WHU-Dataset) and extracted to 
the folder `DeepENF/Data/Hua`. The folders should have the structure

```angular2html
└── Hua
    ├── H0
    ├── H1
    ├── H1_ref
    ├── H1_ref_one_day
```

### 4. Train network
Run the script 
```angular2
DeepENF/Code/train_deepENF.py --enf_est_criterion L1Loss --n_epochs_enf_est 250 --augment --min_snr_in_db -2 --max_snr_in_db 2 --enf_est_n_filters 16 --enf_est_inner_dim 500 --enf_est_n_layers_linear 5 --enf_est_n_layers_cnn 5 --fold 0 --numpy_seed 100 --n_folds 5 --enf_est_kernel_sizes 3 --downsample_fs 800 --lr_estimation 5e-4 --tr_batch_size 256 --val_batch_size 512 --lr_scheduler ReduceLROnPlateau --enf_est_module_type linear_out --nfft_scale 0 --kaiser_bandpass --freq_band_size 0.1 --window_size 1 --n_harmonics_nn 1 --n_harmonics_data 1 --ripple_db 15 --transition_width_hz 0.1
```

### 6. Evaluate network
Run the script 
```angular2
DeepENF/Code/eval_deepENF.py --dataset_type tst --sr_path /media/blue/tsingalis/gitRepositories/DeepENF/experiments/RFA/Fold1/ --single_test --n_epochs_sr 110 --smooth no_smooth --downsampled_fs 800
```


## Reference
If you use this code in your experiments please cite this work by using the following bibtex entry: 

```
@article{someIdToBeUpdated,
  title={DeepENF: A data-driven Electric Network Frequency estimation framework},
  author={Tsingalis, Ioannis and Kotropoulos, Constantine},
  journal={Pattern Recognition Letters},
  year={2025},
  publisher={Elsevier}
}
```