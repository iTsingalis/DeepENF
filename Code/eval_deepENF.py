import os
import json
import numpy as np
import argparse

from Code.utils import load
from Code.ENF import ENF

from pathlib import Path
import matplotlib.pyplot as plt

import ast

import torch
import torchaudio

from utils import print_args

import scipy.signal as ssg
from scipy import stats
from scipy import signal

from tqdm import tqdm

from Code.timer import Timer
from Code.dataset import get_wav_frames

from statistics import mean, stdev, median
import mpltex

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

time = Timer()

linestyles = mpltex.linestyle_generator(lines=['-'])


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def conv_smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def find_best_match_position(lst, window):
    return sum(
        abs(lst[i: i - len(window) + 1 or None] - w)
        for i, w in enumerate(window)
    ).argmin()


def deep_enf(data, sr_module, n_harmonic_to_use=0):
    if not args.hua_rfa:

        # 1. Resample
        data = ssg.resample_poly(data, args.downsampled_fs, 8000)
        # # 2. band pass
        from dataset import filter_signal
        data = filter_signal(data, args)

    # hann is applied here also
    frames, _, nperseg, _ = get_wav_frames(data, args.downsampled_fs,
                                           args.window_size_seconds,
                                           window='hann')

    # nperseg = sr_module.out_features
    hat_ref_frames = []

    if args.nfft_scale:
        nfft = next_power_of_2(int(args.nfft_scale * nperseg))
        bin_size = args.downsampled_fs / nfft
    else:
        nfft = args.downsampled_fs * 2000
        bin_size = args.downsampled_fs / nfft

    freq_bins = np.arange(nfft // 2 + 1) * bin_size

    for i, s in enumerate(tqdm(frames.T, desc='Compute DeepENF...', disable=False, position=0, leave=True)):
        with torch.no_grad():
            hat_ref_frame_tensor = sr_module(torch.from_numpy(s)[None, :].float())
            hat_ref_frame = [h.cpu().numpy().squeeze() for h in hat_ref_frame_tensor]

        if args.rolling_max_frequency and args.window_size > 1:
            # # Rolling frequency
            # hat_ref_frame = uniform_filter1d(hat_ref_frame, size=2)
            max_freq_idx = find_best_match_position(freq_bins, hat_ref_frame)
            max_freq = freq_bins[max_freq_idx]
        else:

            # Iterate every three elements
            max_frequencies = []
            # for n_harmonic, j in enumerate(range(0, len(hat_ref_frame), args.window_size)):
            for n_harmonic, n_harmonics in enumerate(hat_ref_frame, start=1):
                # Middle freq
                assert args.window_size % 2 != 0, 'window length must be odd'
                half_width = args.window_size // 2
                if args.n_harmonics_nn > 1:
                    n_harmonics = [hf / (n_harmonic + 1) for hf in n_harmonics]
                    # max_frequencies.append(group_freq[half_width] / (n_harmonic + 1))
                    max_frequencies.append(n_harmonics[half_width])
                else:
                    if args.window_size > 1:
                        max_frequencies.append(n_harmonics[half_width] / (args.n_harmonics_nn + 1))
                    else:
                        max_frequencies.append(n_harmonics / (args.n_harmonics_nn + 1))

            # return the first harmonic
            max_freq = max_frequencies[n_harmonic_to_use]

        hat_ref_frames.append(max_freq)

    return np.array(hat_ref_frames)


def save_stats(tested_algs, stat_wavs, path, stat_type, base_alg='DeepENF'):
    mse_base = 10 ** -5
    std_base = 10 ** -4

    stat_algs = {k: [] for k in tested_algs}
    for wav_key, stat_dict in stat_wavs.items():
        for stat_alg_key, stat_alg_val in stat_dict.items():
            if stat_type == 'Pearson' or stat_type == 'Spearman':
                stat_algs[stat_alg_key].append((wav_key, stat_alg_val['statistic']))
            elif stat_type == 'MSE':
                stat_algs[stat_alg_key].append((wav_key, stat_alg_val))
            else:
                raise ValueError('Please select stat_type between "MSE", "Pearson", "Spearman"')

    # Sort w.r.t. base algorithm
    wav_key_ordered, stat_base_alg_ordered = list(zip(*sorted(stat_algs[base_alg],
                                                              key=lambda x: x[1], reverse=True)))

    stat_alg_ordered = {k: [] for k in tested_algs}
    for alg in tested_algs:
        stat_alg_ordered[alg] = [dict(stat_algs[alg])[x] if x in dict(stat_algs[alg])
                                 else None for x in list(wav_key_ordered)]

    import matplotlib.ticker as mticker
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for alg in tested_algs:
        ax.semilogy(stat_alg_ordered[alg], label=alg, **next(linestyles))
        ax.set_xlabel('Sorted Reordering Sample Index')
        ax.set_ylabel(stat_type)

        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.set_xticks(np.arange(len(stat_alg_ordered[alg])))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        # ax.legend()
        # plt.show()
    plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
    plt.grid(visible=True, which='minor', color='k', linestyle='-', linewidth=0.2)
    plt.minorticks_on()
    plt.legend()
    plt.savefig(os.path.join(path, f'{stat_type}_sorted.png'), bbox_inches='tight')
    plt.close(fig)

    import csv
    with open(os.path.join(path, f'{stat_type}_{args.dataset_type}_single.csv'), "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['wav_files'] + list(stat_alg_ordered.keys()))
        writer.writerows(zip(*[list(wav_key_ordered)] + list(stat_alg_ordered.values())))

    results = {}
    for alg in tested_algs:
        mean_per_alg_sci = float('{:.4f}'.format(mean(stat_alg_ordered[alg]) / mse_base))
        std_per_alg_sci = float('{:.4f}'.format(stdev(stat_alg_ordered[alg]) / std_base))

        print(f'Alg: {alg} -- mean {stat_type}: {mean_per_alg_sci} x 10^{{-5}}'
              f'-- std {stat_type}: {std_per_alg_sci} x 10^{{-4}} -- '
              f'mean {stat_type}: {mean(stat_alg_ordered[alg])} '
              f'-- std {stat_type}: {stdev(stat_alg_ordered[alg])} '
              f'median {stat_type}: {median(stat_alg_ordered[alg])} ')

        results.update({alg: {
            f'std {stat_type} sci': f'{std_per_alg_sci} x 10^{{-4}}',
            f'mean {stat_type} sci': f'{mean_per_alg_sci} x 10^{{-5}}',
            f'std {stat_type}': stdev(stat_alg_ordered[alg]),
            f'mean {stat_type}': mean(stat_alg_ordered[alg]),
            f'median {stat_type}': median(stat_alg_ordered[alg]),
        }})

    with open(os.path.join(path, f'{stat_type}_mean_std_single.json'), 'w', encoding='utf8') as fp:
        json.dump(results, fp, indent=4)


def single_test(data_pair, ref_folder, target_folder, snr=None):
    save_flag = False

    n_samples = None
    # tested_algs = ['Hua', 'Welch', 'MUSIC', 'STFT', 'DeepENF']
    tested_algs = ['Hua', 'DeepENF']

    mse_wavs = dict([(target_key, {k: None for k in tested_algs}) for target_key, _ in data_pair[:n_samples]])
    pearson_wavs = dict(
        [(target_key, {k: {'statistic': None, 'pvalue': None} for k in tested_algs}) for target_key, _ in
         data_pair[:n_samples]])
    spearman_wavs = dict(
        [(target_key, {k: {'statistic': None, 'pvalue': None} for k in tested_algs}) for target_key, _ in
         data_pair[:n_samples]])

    tested_algs_dict = {k: None for k in tested_algs}

    path = os.path.join(args.sr_path, 'enf', f'{args.n_epochs_sr}')

    Path(path).mkdir(parents=True, exist_ok=True)

    print('Load DeepENF model...')
    sr_module, _, _, _, _ = load(os.path.join(args.sr_path, f'final_model_e{args.n_epochs_sr}.pth'),
                                 os.path.join(args.sr_path, f'tuned_lr.json'),
                                 module_type='estimation')
    sr_module.cpu()
    sr_module.eval()

    normalize_signals = False
    print('Start evaluation...')

    # for target_file_name, ref_file_name in tqdm(data_pair[:n_samples], desc='DeepENF wav...',
    #                                             disable=False):

    # ('038.wav', '038_ref.wav') # fold 0
    # ('111.wav', '111_ref.wav') # fold 1
    # ('088.wav', '088_ref.wav') # fold 4
    for target_file_name, ref_file_name in tqdm([('111.wav', '111_ref.wav')], desc='DeepENF wav...',
                                                disable=False):

        print('Load wav files..')
        #############################
        # Load reference file sample
        #############################
        ref_data, ref_fs = torchaudio.load(os.path.join(ref_folder, ref_file_name))
        ref_data = ref_data.squeeze()

        # ##########################
        # # Load target file sample
        # ##########################
        target_data, target_fs = torchaudio.load(os.path.join(target_folder, target_file_name))
        target_data = target_data.squeeze()

        #################################
        # Compute ENF in reference data using STFT.
        #################################
        print('Compute STFT for reference data')
        nominal_freq = 50

        ref_enf = ENF(synthetic_fs=ref_fs,
                      downsampled_fs=None,
                      nominal_freq=nominal_freq,
                      freq_band_size=args.freq_band_size,
                      window_size_seconds=args.window_size_seconds,
                      harmonic_n=1,
                      nfft_scale=args.nfft_scale,
                      normalize_signals=normalize_signals,
                      single_bandpass=False)

        ref_enf_output = ref_enf.fit_enf(ref_data)

        # multiply by 2 to compute the target and reference statistics in second harmonic
        ref_enf = 2 * np.array(ref_enf_output['enf'])

        hua_outputs = '/media/blue/tsingalis/gitRepositories/DeepENF/Data/hua_outputs/'

        # hua_path = f'{hua_outputs}/noSNR/'
        hua_path = f'{hua_outputs}/'

        with open(os.path.join(hua_path, 'f_ref', f'{target_file_name}.txt'), "r") as infile:
            hua_ref_enf = np.array([float(line.rstrip('\n')) for line in infile])

        # ##############################
        # # Compute ENF in target data.
        # ##############################
        for tested_alg_name, _ in tested_algs_dict.items():
            if tested_alg_name.lower() == 'DeepENF'.lower():

                enf_output = deep_enf(data=target_data,
                                      sr_module=sr_module,
                                      n_harmonic_to_use=0)

                enf_output *= 2

            elif tested_alg_name.lower() == 'Hua'.lower():
                if snr is not None:
                    raise ValueError('Not supported yet.')
                    # hua_path = f'{hua_outputs}/SNR{snr}/'
                else:
                    hua_path = f'{hua_outputs}/'

                with open(os.path.join(hua_path, 'f_P_MLE_ENF', f'{target_file_name}.txt'), "r") as infile:
                    enf_output = [float(line.rstrip('\n')) for line in infile]

            else:
                # print(f'Compute {tested_alg_name} for target data')
                obj_enf = ENF(synthetic_fs=target_fs,
                              downsampled_fs=args.downsampled_fs,
                              nominal_freq=nominal_freq,
                              freq_band_size=args.freq_band_size,
                              window_size_seconds=args.window_size_seconds,
                              harmonic_n=2,
                              nfft_scale=args.nfft_scale,
                              method=tested_alg_name.lower(),
                              normalize_signals=normalize_signals)

                obj_enf_fitted = obj_enf.fit_enf(target_data)
                enf_output = 2 * np.array(obj_enf_fitted['enf'])

            # multiply by 2 to compute the target and reference statistics in second harmonic
            tested_algs_dict[tested_alg_name] = enf_output

        print_str = []
        for tested_alg_name, _ in tested_algs_dict.items():
            if tested_alg_name.lower() == 'Hua'.lower():
                mse_per_alg = np.linalg.norm(tested_algs_dict[tested_alg_name] - hua_ref_enf) ** 2 / len(ref_enf)
            else:
                mse_per_alg = np.linalg.norm(tested_algs_dict[tested_alg_name] - ref_enf) ** 2 / len(ref_enf)

            mse_wavs[target_file_name][tested_alg_name] = mse_per_alg

            pearson_per_alg = stats.pearsonr(tested_algs_dict[tested_alg_name], ref_enf)
            pearson_wavs[target_file_name][tested_alg_name]['statistic'] = pearson_per_alg.statistic
            pearson_wavs[target_file_name][tested_alg_name]['pvalue'] = pearson_per_alg.pvalue

            spearman_per_alg = stats.spearmanr(a=tested_algs_dict[tested_alg_name], b=ref_enf)
            spearman_wavs[target_file_name][tested_alg_name]['statistic'] = spearman_per_alg.statistic
            spearman_wavs[target_file_name][tested_alg_name]['pvalue'] = spearman_per_alg.pvalue

            mse_per_alg_sci = float('{:.4f}'.format(mse_per_alg / 10 ** -5))
            print_str.append(
                f'{target_file_name} {tested_alg_name} {mse_per_alg_sci} x 10^{{-5}} ({mse_per_alg})'
            )
        print(', '.join(print_str))

        import pandas as pd

        df = pd.DataFrame.from_dict(tested_algs_dict, orient='index').T
        df['Reference'] = ref_enf

        import matplotlib.ticker as mticker

        fig, ax = plt.subplots(nrows=1, ncols=1)
        bplot = df[['Hua', 'DeepENF', 'Reference']].boxplot(['Hua', 'DeepENF', 'Reference'], figsize=(10, 10), showfliers=True)
        bplot.set_xticks([1, 2, 3], ['P-MLE', 'E-DeepENF', 'Reference'])
        plt.yticks(rotation=45, ha='right')
        bplot.set_ylabel('Frequency (Hz)', size=14)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        plt.xticks(size=14)
        plt.yticks(size=14)

        # plt.ticklabel_format(axis='y', style='sci', scilimits=(-5, -5))
        # plt.savefig(os.path.join('./', f'{stat_type}_single_hua_box_plot.png'), bbox_inches='tight', pad_inches=0.1)
        # plt.savefig(os.path.join('./', f'{stat_type}_single_hua_box_plot.eps'),
        #             bbox_inches='tight', pad_inches=0.1, format='eps')

        # if save_flag:
        plt.savefig(os.path.join(path, f'enf_{target_file_name}_box.png'), bbox_inches='tight')
        plt.savefig(os.path.join(path, f'enf_{target_file_name}_box.eps'), bbox_inches='tight', format='eps')
        plt.close(fig)
        # else:
        #     plt.show()

        if save_flag:
            df.to_csv(os.path.join(path, f'enf_{target_file_name}.csv'), index=False)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_tight_layout(True)
        # ax.plot(ref_enf, label='Reference')
        ax.plot(ref_enf, label='Reference')
        # ax.plot(hua_ref_enf, label='Matlab Reference')

        ax.ticklabel_format(useOffset=False)

        # plot the ENF of th remaining algorithms
        for tested_alg_name, _ in tested_algs_dict.items():
            if tested_alg_name == 'Hua' or tested_alg_name == 'DeepENF':
                if tested_alg_name == 'Hua':
                    ax.plot(np.array(tested_algs_dict[tested_alg_name]), label='P-MLE')
                else:
                    ax.plot(np.array(tested_algs_dict[tested_alg_name]),
                            label='E-' + tested_alg_name)
            # disable sci notation
            ax.ticklabel_format(useOffset=False)
            ax.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Time Frames', size=14)
            ax.set_ylabel('Frequency (Hz)', size=14)
        plt.legend(prop={'size': 14})
        plt.xticks(size=14)
        plt.yticks(size=14)

        # plt.grid()
        # if save_flag:
        plt.savefig(os.path.join(path, f'enf_{target_file_name}.png'), bbox_inches='tight')
        plt.savefig(os.path.join(path, f'enf_{target_file_name}.eps'), bbox_inches='tight', format='eps')
        plt.close(fig)
        # else:
        #     plt.show()

        if save_flag:
            with open(os.path.join(path, f'mse_per_wav.json'), 'w', encoding='utf8') as fp:
                json.dump(mse_wavs, fp, indent=4)

    if save_flag:
        with open(os.path.join(path, f'pearson_per_wav.json'), 'w', encoding='utf8') as fp:
            json.dump(pearson_wavs, fp, indent=4)

        with open(os.path.join(path, f'spearman_per_wav.json'), 'w', encoding='utf8') as fp:
            json.dump(spearman_wavs, fp, indent=4)

    if save_flag:
        base_alg = 'Hua'
        save_stats(tested_algs, mse_wavs, path=path, stat_type='MSE', base_alg=base_alg)
        save_stats(tested_algs, pearson_wavs, path=path, stat_type='Pearson', base_alg=base_alg)
        save_stats(tested_algs, spearman_wavs, path=path, stat_type='Spearman', base_alg=base_alg)


def main():

    # Change this according to your folders
    hua_root_folder = "/media/blue/tsingalis/DeepENF/Data/Hua/"

    if args.hua_rfa:
        target_folder = f"{hua_root_folder}/H1_rfa/"
    else:
        target_folder = f"{hua_root_folder}/H1/"

    ref_folder = f"{hua_root_folder}/H1_ref/"

    with open(os.path.join(folds_dir, f'folds{args.n_splits}', f"tr_data_pair_fold_{args.fold}.txt"), 'r') as f:
        tr_data_pair = ast.literal_eval(f.read())

    with open(os.path.join(folds_dir, f'folds{args.n_splits}', f"val_data_pair_fold_{args.fold}.txt"), 'r') as f:
        val_data_pair = ast.literal_eval(f.read())

    with open(os.path.join(folds_dir, f'folds{args.n_splits}', f"tst_data_pair_fold_{args.fold}.txt"), 'r') as f:
        tst_data_pair = ast.literal_eval(f.read())

    n_samples = None
    if args.dataset_type == 'train':
        data_pair = tr_data_pair
    elif args.dataset_type == 'val':
        data_pair = val_data_pair
    elif args.dataset_type == 'tst':
        data_pair = tst_data_pair

    data_pair = data_pair[:n_samples]

    time.start()
    single_test(data_pair, ref_folder, target_folder)
    time.stop(tag='single_test', verbose=True)


if __name__ == '__main__':
    """
--dataset_type
tst
--sr_path
/media/blue/tsingalis/gitRepositories/DeepENF/experiments/RFA/Fold4/
--single_test
--n_epochs_sr
110
--smooth
no_smooth
--downsampled_fs
800
    """
    parser = argparse.ArgumentParser(description='DeepENF')

    parser.add_argument('--dataset_type',
                        default=None, type=str, required=True, choices=['train', 'val', 'tst'],
                        help='Compute mse on train or val set.')

    parser.add_argument('--smooth',
                        default='no_smooth', type=str, required=False,
                        choices=['conv', 'savgol_filter', 'conv_filter', 'no_smooth'],
                        help='What type of smooth to apply on the output ENF.')

    parser.add_argument('--sr_path', default=None, type=str, required=True,
                        help='ENF-representation module path.')

    parser.add_argument('--nfft_scale', type=int, default=0)
    parser.add_argument('--folds_dir', type=str,
                        default='./checkpoint/experiment_name', help='output directory')

    # Data
    parser.add_argument('--audio_length', type=int, default=133623)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', '-r', action='store_true')

    parser.add_argument('--fold', type=int, required=False,
                        choices=[0, 1, 2, 3, 4])

    parser.add_argument('--n_splits', type=int, required=False, default=5,
                        choices=[5, 10])

    # Training
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--log_frequency', type=int, default=25)
    parser.add_argument('--save_frequency', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)

    parser.add_argument('--sr_n_filters', type=int, default=16,
                        help='number of filters per layer in the enf representation (sr) module')
    parser.add_argument('--sr_inner_dim', type=int, default=1800,
                        help='dimension after first linear transformation')
    parser.add_argument('--sr_kernel_size', type=int, default=5,
                        help='filter size in the convolutional blocks of the enf representation (sr) module')

    parser.add_argument('--lr_estimation', type=float, default=1e-4,
                        help='initial learning rate for adam optimizer used for the enf-representation module')
    parser.add_argument('--lr_detection', type=float, default=0.0003,
                        help='initial learning rate for adam optimizer used for the frequency-detection module')

    parser.add_argument('--n_epochs_fd', type=int, default=300,
                        help='number of epochs used to train the frequency detection (fd) module')
    parser.add_argument('--n_epochs_sr', type=int, default=250,
                        help='number of epochs used to train the enf representation (fr) module')

    parser.add_argument('--sr_criterion', type=str, default='MSELoss',
                        choices=["SmoothL1Loss", "MSELoss", "L1Loss"],
                        help='The criterion used to train the enf representation module')

    parser.add_argument('--snr', type=float, default=None, help='snr parameter')

    parser.add_argument('--window_size_seconds', type=int, default=10, help='Window size in seconds.')

    parser.add_argument('--downsampled_fs', type=int,
                        default=8000,
                        help='The sampling frequency of target signal.')

    parser.add_argument('--target_fs', type=int,
                        default=8000,
                        help='The sampling frequency of target signal.')

    parser.add_argument('--rolling_window_size_samples', type=int, default=20,
                        help='Window size in samples.')

    parser.add_argument('--rolling_mean_repetitions', type=int, default=1,
                        help='Number of times to apply the mean rolling window.')

    parser.add_argument('--load_json',
                        help='Load settings from file in json format. Command line options override values in file.')

    parser.add_argument('--single_test', action='store_true', help="Apply no snr test.")
    parser.add_argument('--snr_test', action='store_true', help="Apply snr test")

    parser.add_argument('--music_enf', action='store_true', help="Apply MUSIC for enf estimation")

    parser.add_argument('--enf_est_module_type', type=str,
                        default='linear_out', help='type of the fr module: [linear_out | conv_trans_out]')

    parser.add_argument('--freq_band_size', type=int, default=0.1, required=False,
                        help='The frequency band for the bandpass filter')

    parser.add_argument('--window_size', type=int, default=1,
                        help='The number of the frequencies in the window to keep.')

    parser.add_argument('--transition_width_hz', type=float, default=0.1, required=False,
                        help='The transition width in Hz for the bandpass filter')

    parser.add_argument('--ripple_db', type=float, default=15, required=False,
                        help='The riple in Db for the bandpass filter')

    parser.add_argument('--rolling_max_frequency', action='store_true',
                        help="Compute max frequency using rolling window")

    args = parser.parse_args()

    with open(os.path.join(args.sr_path, 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    print_args(args)

    # Change this according to your folders
    folds_dir = '/media/blue/tsingalis/gitRepositories/DeepENF/Data/Folds'

    main()
