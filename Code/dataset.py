import ast

import numpy as np

from scipy import signal
import scipy.signal as ssg
from scipy.signal import butter, sosfreqz, sosfilt
from scipy.signal._spectral_py import _triage_segments
from scipy.signal import kaiserord, firwin, filtfilt

from scipy.fft import rfft
from pathlib import Path
from tqdm import tqdm

import os
import torch
import torchaudio
import torch.utils.data as data

AUDIO_EXTENSIONS = ['.mp3', '.wav']


def target_ref_pair(target_folder, ref_folder):
    ref_files = [Path(f).stem for f in os.listdir(ref_folder) if f.endswith('.wav')]
    target_files = [Path(f).stem for f in os.listdir(target_folder) if f.endswith('.wav')]

    target_ref_pair_dict = {}
    for target_file in target_files:
        try:
            ref_file = [f for f in ref_files if target_file in f].pop()
        except IndexError as e:
            print(f'Error: {e} -- target file {target_file} is not in ref files.')
            continue

        target_ref_pair_dict.update({target_file: ref_file})

    _target_ref_pair = [(str(k).zfill(3) + '.wav', target_ref_pair_dict[k] + '.wav')
                        for k in sorted(target_ref_pair_dict, key=lambda tt: (int(tt), tt))]

    return _target_ref_pair


def target_ref_one_day_pair(target_folder, ref_one_day_folder):
    if ref_one_day_folder is None:
        return None

    target_files = [Path(f).stem for f in os.listdir(target_folder) if f.endswith('.wav')]
    ref_one_day_files = [Path(f).stem for f in os.listdir(ref_one_day_folder) if f.endswith('.wav')]

    target_ref_one_day_pair_dict = {}
    for target_file in target_files:
        for f in ref_one_day_files:
            try:
                x, y, _ = f.replace('_', '-').split('-')
                x, y = int(x), int(y)
                if x <= int(target_file) <= y:
                    target_ref_one_day_pair_dict.update({target_file: f})
            except ValueError:
                x, _ = f.replace('_', '-').split('-')
                x = int(x)
                if x == int(target_file):
                    target_ref_one_day_pair_dict.update({target_file: f})

    _target_ref_one_day_pair = [(str(k).zfill(3) + '.wav', target_ref_one_day_pair_dict[k] + '.wav')
                                for k in sorted(target_ref_one_day_pair_dict, key=lambda tt: (int(tt), tt))]

    return _target_ref_one_day_pair


def read_wav_pairs(target_folder, ref_folder, ref_one_day_folder=None):
    _target_ref_pair = target_ref_pair(target_folder, ref_folder)

    _target_ref_one_day_pair = target_ref_one_day_pair(target_folder, ref_one_day_folder)

    return _target_ref_pair, _target_ref_one_day_pair


class AudioFolder(torch.utils.data.Dataset):
    def __init__(self, ref_frames, ref_frames_freq, trg_frames, n_harmonics_nn):
        self.trg_frames = trg_frames
        self.ref_frames = ref_frames

        self.ref_frames_freq = ref_frames_freq
        self.n_harmonics_nn = n_harmonics_nn

        self.reference_frame_size, self.reference_frame_dim = ref_frames.shape
        self.target_signal_size, self.target_signal_dim = trg_frames.shape

    def __getitem__(self, index):
        return (self.trg_frames[index],
                self.ref_frames[index],
                [self.ref_frames_freq[h][index] for h in range(self.n_harmonics_nn)])

    def __len__(self):
        return self.target_signal_size


def get_loaders(args, target_folder, ref_folder,
                tr_n_samples=None,
                val_n_samples=None,
                tst_n_samples=None,
                folds_dir=None):
    tr_path = os.path.join(folds_dir, f'folds{args.n_folds}', f"tr_data_pair_fold_{args.fold}.txt")
    print(f'Load tr: {tr_path}')
    with open(tr_path, 'r') as f:
        tr_data_pair = ast.literal_eval(f.read())

    val_path = os.path.join(folds_dir, f'folds{args.n_folds}', f"val_data_pair_fold_{args.fold}.txt")
    print(f'Load val: {val_path}')
    with open(val_path, 'r') as f:
        val_data_pair = ast.literal_eval(f.read())

    tst_path = os.path.join(folds_dir, f'folds{args.n_folds}', f"tst_data_pair_fold_{args.fold}.txt")
    print(f'Load tst: {tst_path}')
    with open(tst_path, 'r') as f:
        tst_data_pair = ast.literal_eval(f.read())

    # Get train frames
    tr_frames = wav_frames(args,
                           tr_data_pair,
                           target_folder,
                           ref_folder,
                           n_samples=tr_n_samples,
                           desc='Prepare training dataset...')

    tr_ref_frames, tr_ref_frames_freq, tr_trg_frames, trg_fs, ref_fs = tr_frames

    # Get validation frames
    val_frames = wav_frames(args,
                            val_data_pair,
                            target_folder,
                            ref_folder,
                            n_samples=val_n_samples,
                            desc='Prepare validation dataset...')

    val_ref_frames, val_ref_frames_freq, val_trg_frames, trg_fs, ref_fs = val_frames

    # Get test frames
    tst_frames = wav_frames(args,
                            tst_data_pair,
                            target_folder,
                            ref_folder,
                            n_samples=tst_n_samples,
                            desc='Prepare test dataset...')

    tst_ref_frames, tst_ref_frames_freq, tst_trg_frames, trg_fs, ref_fs = tst_frames

    tr_dataset = AudioFolder(tr_ref_frames, tr_ref_frames_freq, tr_trg_frames, args.n_harmonics_nn)
    del tr_trg_frames
    del tr_ref_frames

    val_dataset = AudioFolder(val_ref_frames, val_ref_frames_freq, val_trg_frames, args.n_harmonics_nn)
    del val_ref_frames
    del val_trg_frames

    tst_dataset = AudioFolder(tst_ref_frames, tst_ref_frames_freq, tst_trg_frames, args.n_harmonics_nn)
    del tst_ref_frames
    del tst_trg_frames

    train_loader = torch.utils.data.DataLoader(tr_dataset,
                                               batch_size=args.tr_batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               shuffle=True)

    validation_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=args.val_batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)

    test_loader = torch.utils.data.DataLoader(tst_dataset,
                                              batch_size=args.tst_batch_size,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    print(f'Train size: {len(train_loader.dataset)} '
          f'-- Validation size: {len(validation_loader.dataset)} '
          f'-- Test size: {len(test_loader.dataset)}')

    print(f'Original Input signal dim {train_loader.dataset.target_signal_dim} '
          f'-- Original Output signal dim {train_loader.dataset.reference_frame_dim}')

    loaders = train_loader, validation_loader, test_loader
    # signals_dim = tr_dataset.target_signal_dim, tr_dataset.reference_frame_dim, tr_dataset.reference_spectrum_dim
    signals_dim = tr_dataset.target_signal_dim, tr_dataset.reference_frame_dim

    signals_fs = trg_fs, ref_fs
    return loaders, signals_dim, signals_fs


def get_max_freq(frames, data_fs, nperseg, freq_band_size,
                 nfft_scale, n_harmonics, window_size=1):
    if nfft_scale:
        nfft = next_power_of_2(int(nfft_scale * nperseg))
        bin_size = data_fs / nfft
    else:
        nfft = data_fs * 2000
        bin_size = data_fs / nfft

    freq_bins = np.arange(nfft // 2 + 1) * bin_size

    Zxx = np.apply_along_axis(rfft, arr=frames, axis=0, n=nfft)

    # Do not ignore the first harmonic on 50 Hz.
    freq_ranges = [(n * (50 - freq_band_size), n * (50 + freq_band_size))
                   for n in range(1, n_harmonics + 1)]

    _max_freqs = []
    # Iterate through the specified frequency ranges.
    for n_harmonic, freq_range in enumerate(freq_ranges, start=1):
        start_freq, end_freq = freq_range

        # Find the indices of the frequencies within the specified range
        indices_in_freq_range = np.where((freq_bins >= start_freq) & (freq_bins <= end_freq))[0]

        # Find the maximum values per frame within a specified range of frequencies
        top_indices_in_freq_range = np.argmax(np.abs(Zxx[indices_in_freq_range]), axis=0)
        # Get the index of the maximum freq per frame
        top_index_per_frame = indices_in_freq_range[top_indices_in_freq_range]

        window_freq_per_frame = []
        # Iterate over the frames
        for max_freq_idx in top_index_per_frame:
            half_width = window_size // 2
            start = max_freq_idx - half_width
            end = max_freq_idx + half_width + 1
            # get the max freq and the surrounding frequencies in a frame using a window size
            frame_window_freq = freq_bins[start:end]

            # Threshold to avoid extreme values
            lower_threshold = n_harmonic * 50 - freq_band_size
            upper_threshold = n_harmonic * 50 + freq_band_size

            frame_window_freq = np.where(frame_window_freq < lower_threshold, lower_threshold, frame_window_freq)
            frame_window_freq = np.where(frame_window_freq > upper_threshold, upper_threshold, frame_window_freq)

            window_freq_per_frame.append(frame_window_freq.astype(np.float32))

        # append info about each harmonic
        _max_freqs.append(np.vstack(window_freq_per_frame))

    return _max_freqs


def frame(data, nperseg, noverlap, equal_sized=False):
    """
        https://github.com/dpwe/audfprint/blob/cb03ba99feafd41b8874307f0f4e808a6ce34362/stft.py
    """
    step = nperseg - noverlap
    num_samples = data.shape[0]
    num_frames = 1 + ((num_samples - nperseg) // step)
    shape = (num_frames, nperseg) + data.shape[1:]
    if equal_sized:
        result = np.zeros(shape)
        window_pos = range(0, len(data) - nperseg + 1, step)
        for i, w in enumerate(window_pos):
            result[i] = data[w:w + nperseg]
    else:

        strides = (data.strides[0] * step,) + data.strides
        result = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    return result.T, num_frames


def get_wav_frames(data, data_fs, window_size_seconds, window='hann'):
    nperseg = data_fs * window_size_seconds  # Length of each segment.
    noverlap = data_fs * (window_size_seconds - 1)

    window_vector, _nperseg = _triage_segments(window, nperseg, input_length=len(data))
    assert _nperseg == nperseg

    frames, n_frames = frame(data, nperseg, noverlap, equal_sized=True)

    frames = frames * window_vector[:, np.newaxis]

    return frames, n_frames, nperseg, noverlap


def butter_bandpass_filter(data, locut, hicut, synthetic_fs, order):
    """Passes input data through a Butterworth bandpass filter. Code borrowed from
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

    :param data: list of signal sample amplitudes
    :param locut: frequency (in Hz) to start the band at
    :param hicut: frequency (in Hz) to end the band at
    :param synthetic_fs: the sample rate
    :param order: the filter order
    :returns: list of signal sample amplitudes after filtering
    """
    nyq = 0.5 * synthetic_fs
    low = locut / nyq
    high = hicut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')

    plt_freq_response = False
    if plt_freq_response:
        import matplotlib.pyplot as plt
        from scipy.signal import sosfreqz

        # Plot the frequency response
        plt.figure(figsize=(10, 6))

        w, h = sosfreqz(sos, worN=2000)
        # plt.plot(0.5 * synthetic_fs * w / np.pi, 20*np.log10(np.abs(h)))
        plt.plot(0.5 * synthetic_fs * w / np.pi, 20 * np.log10(np.abs(h)))

        plt.title('Band-pass Butter Filter Frequency Response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        plt.show()

    return signal.sosfilt(sos, data)


def bandpass_filter(frame, enf_harmonic_n, signal_fs, freq_band_size=0.1, nominal_freq=50):
    ref_locut = enf_harmonic_n * (nominal_freq - freq_band_size)
    ref_hicut = enf_harmonic_n * (nominal_freq + freq_band_size)
    frame = butter_bandpass_filter(frame, ref_locut, ref_hicut, signal_fs, order=10).astype(np.float32)
    return frame


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def bandpass_kaiser(cut_off, fs, transition_width_hz=5.0, ripple_db=25.0):
    # Nyquist rate.
    nyq_rate = fs / 2

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = transition_width_hz / nyq_rate

    n_taps, beta = kaiserord(ripple_db, width=width)

    if n_taps % 2 == 0:
        n_taps = n_taps + 1

    # Estimate the filter coefficients.
    cut_off_nyq = [cof / nyq_rate for cof in cut_off]

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    # See example why we need: pass_zero=False or 'bandpass'
    taps = firwin(n_taps, cut_off_nyq, window=('kaiser', beta), pass_zero=False)

    return taps


def bandpass_firwin(n_taps, cut_off, fs, transition_width_hz=5.,
                    ripple_db=25, window='hamming'):
    if window == 'hamming' or window == 'blackman':
        taps = firwin(n_taps, cutoff=cut_off, fs=fs, pass_zero=False,
                      window=window, scale=False)
    elif window == 'kaiser':
        taps = bandpass_kaiser(cut_off, fs, transition_width_hz, ripple_db)
    else:
        raise ValueError('window should be hamming or kaiser')

    return taps


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_multi_bandpass_filter(data, synthetic_fs, n_harmonics_data, freq_band_size, order=10):
    cut_offs = []
    for n in range(1, n_harmonics_data + 1):  # Start from the second harmonic 100Hz
        cut_offs.append([(n + 1) * (50 - freq_band_size),
                         (n + 1) * (50 + freq_band_size)])

    sos_list = [butter_bandpass(lowcut, highcut, synthetic_fs, order=order) for lowcut, highcut in cut_offs]

    plt_freq_response = False
    if plt_freq_response:
        import matplotlib.pyplot as plt
        # Plot the frequency response
        plt.figure(figsize=(10, 6))
        for i in range(len(cut_offs)):
            w, h = sosfreqz(sos_list[i], worN=2000)
            plt.plot(0.5 * synthetic_fs * w / np.pi, 20 * np.log10(np.abs(h)),
                     label=f'Band {i + 1}: {cut_offs[i]} Hz')

        plt.title('Multiband Filter Frequency Response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain')
        plt.legend()
        plt.xlim(0, 250)
        plt.grid(True)
        plt.show()

    # Apply the multi-band filter to the data
    return np.sum([sosfilt(sos, data) for sos in sos_list], axis=0)


def filter_signal(data, args):
    if args.butter_bandpass:
        # Apply bandpass in data
        filtered_data = bandpass_filter(data, enf_harmonic_n=2,
                                        signal_fs=args.downsample_fs,
                                        freq_band_size=args.freq_band_size,
                                        nominal_freq=50)
    elif args.butter_multi_bandpass:

        filtered_data = butter_multi_bandpass_filter(data, args.downsample_fs,
                                                     args.n_harmonics_data,
                                                     args.freq_band_size,
                                                     order=10)

    elif args.kaiser_bandpass:

        cut_offs = []
        for n in range(1, args.n_harmonics_data + 1):  # Start from the second harmonic 100Hz
            # cut_offs.extend([(n + 1) * (50 - args.freq_band_size),
            #                  (n + 1) * (50 + args.freq_band_size)])
            cut_offs.extend([(n + 1) * (50 - args.freq_band_size),
                             (n + 1) * (50 + args.freq_band_size)])

        taps_kaiser = bandpass_kaiser(cut_off=cut_offs, fs=args.downsample_fs,
                                      transition_width_hz=args.transition_width_hz,
                                      ripple_db=args.ripple_db)

        b, a = taps_kaiser, [1]
        filtered_data = filtfilt(b=b, a=1, axis=0, x=data,
                                 padtype='odd',
                                 padlen=3 * (max(len(b), len(a)) - 1))

    plt_spectrum = False
    if plt_spectrum:
        N = len(filtered_data)
        Y = rfft(filtered_data, n=N)
        import matplotlib.pyplot as plt
        plt.plot(np.arange(N / 2) / N * args.downsample_fs, 20 * np.log10(abs(Y)), 'g-', label='FFT filtered signal')
        plt.ylabel(r'Power Spectrum (dB)', fontsize=8)
        plt.xlabel("frequency (Hz)", fontsize=8)
        plt.grid()
        plt.legend(loc='upper right')
        plt.show()

    return filtered_data


def wav_frames(args,
               target_ref_pair,
               target_folder, ref_folder,
               n_samples=None,
               desc=None):
    target_frames, reference_frames, target_frames_freq = [], [], []

    reference_frames_freq = [[] for __ in range(args.n_harmonics_nn)]

    for target_file_name, ref_file_name in tqdm(target_ref_pair[:n_samples], desc=desc):
        #############################
        # Load reference file sample
        #############################
        reference_data, ref_fs = torchaudio.load(os.path.join(ref_folder, ref_file_name))
        reference_data = np.squeeze(reference_data.numpy())

        ##########################
        # Load target file sample
        ##########################
        target_data, _ = torchaudio.load(os.path.join(target_folder, target_file_name))
        target_data = np.squeeze(target_data.numpy())

        if not args.hua_rfa:
            target_data = ssg.resample_poly(target_data, args.downsample_fs, 8000)

            # Filter target signal in specific bands.
            target_data = filter_signal(target_data, args)

        _trg_frames, _, _, _ = get_wav_frames(target_data,
                                              args.downsample_fs,
                                              args.window_size_seconds,
                                              window='hann')

        _reference_frames, _, _reference_nperseg, _ = get_wav_frames(reference_data, ref_fs,
                                                                     args.window_size_seconds,
                                                                     'boxcar')

        # We do NOT filter the Reference ENF Signal.
        # We just look at the specific frequency part of the signal.
        # Alternatively we can filter the signal.
        # However, we do not do that because we consider the signal as a clean signal.
        _reference_frames_freq = get_max_freq(_reference_frames, ref_fs, _reference_nperseg,
                                              freq_band_size=args.freq_band_size,
                                              nfft_scale=args.nfft_scale,
                                              n_harmonics=args.n_harmonics_nn,
                                              window_size=args.window_size)

        target_frames.append(_trg_frames.astype(np.float32))
        reference_frames.append(_reference_frames.astype(np.float32))

        # reference_frames_freq.append(_reference_frames_freq.astype(np.float32))

        for i in range(args.n_harmonics_nn):
            reference_frames_freq[i].extend(_reference_frames_freq[i])

    reference_frames = np.hstack(reference_frames).T
    target_frames = np.hstack(target_frames).T

    return reference_frames, reference_frames_freq, target_frames, args.downsample_fs, ref_fs
