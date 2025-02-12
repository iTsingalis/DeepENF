
import numpy as np
import scipy
from scipy import signal
from scipy.fft import rfft
import scipy.signal as ssg
from scipy.signal._spectral_py import _triage_segments
from scipy.signal import filtfilt, firwin, kaiserord



def quadratic_interpolation(data, max_idx, bin_size, freq_band_size=0.1, enf_harmonic_n=1, threshold=False):
    """
        https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    """
    left = data[max_idx - 1]
    center = data[max_idx]
    right = data[max_idx + 1]

    if threshold:
        peak = (max_idx) * bin_size
    else:
        p = 0.5 * (left - right) / (left - 2 * center + right)
        peak = (max_idx + p) * bin_size  # interpolated peak

    if peak < enf_harmonic_n * (50 - freq_band_size):
        peak = enf_harmonic_n * (50 - freq_band_size)
    elif peak > enf_harmonic_n * (50 + freq_band_size):
        peak = enf_harmonic_n * (50 + freq_band_size)

    return peak


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
    taps = firwin(n_taps, cut_off_nyq,
                  window=('kaiser', beta),
                  pass_zero=False)

    return taps


# https://github.com/dpwe/audfprint/blob/cb03ba99feafd41b8874307f0f4e808a6ce34362/stft.py
def frame(data, nperseg, noverlap, equal_sized=False):
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


class ENF:

    def __init__(self, synthetic_fs,
                 nominal_freq,
                 freq_band_size,
                 window_size_seconds,
                 downsampled_fs=None,
                 harmonic_n=1,
                 nfft_scale=0,
                 tau_lim=1500,
                 normalize_signals=False,
                 multi_bandpass_kaiser=False,
                 single_bandpass=False,
                 multi_bandpass_butter=False,
                 fm=None,
                 method='stft'):

        self.multi_bandpass_butter = multi_bandpass_butter
        self.synthetic_fs = synthetic_fs
        self.nominal_freq = nominal_freq
        self.freq_band_size = freq_band_size
        self.window_size_seconds = window_size_seconds
        self.harmonic_n = harmonic_n
        self.nfft_scale = nfft_scale
        self.downsampled_fs = downsampled_fs
        self.method = method
        self.normalize_signals = normalize_signals
        self.multi_bandpass_kaiser = multi_bandpass_kaiser
        self.single_bandpass = single_bandpass
        self.fm = fm
        self.tau_lim = tau_lim
        assert method.lower() in ['music', 'esprit', 'stft', 'welch']

        assert not (
                single_bandpass and
                multi_bandpass_kaiser and
                multi_bandpass_butter
        ), 'Only single_bandpass or multi_bandpass_kaiser can be True'

        if downsampled_fs is None:
            self.downsampled_fs = synthetic_fs

    def butter_bandpass_filter(self, data, locut, hicut, synthetic_fs, order):
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

        return signal.sosfilt(sos, data)

    def next_power_of_2(self, x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def spectrum_estimation(self, data, synthetic_fs, window_size_seconds=16., window_type='hann'):

        nperseg = synthetic_fs * window_size_seconds  # Length of each segment.
        noverlap = synthetic_fs * (window_size_seconds - 1)

        assert scipy.signal.check_COLA(window_type, nperseg, noverlap, tol=1e-10), 'COLA criterion not met'

        if self.nfft_scale:
            nfft = self.next_power_of_2(int(self.nfft_scale * nperseg))
            bin_width = synthetic_fs / nfft
            threshold = False
        else:
            nfft = synthetic_fs * 2000
            bin_width = synthetic_fs / nfft
            threshold = True

        window_vector, nperseg = _triage_segments(window_type, nperseg, input_length=len(data))

        frames, n_frames = frame(data, nperseg, noverlap, equal_sized=True)

        hop_size = nperseg - noverlap
        t = np.arange(nperseg / 2, nperseg / 2 + n_frames * hop_size, hop_size) / synthetic_fs
        f = np.arange(nfft // 2 + 1) * bin_width

        if self.method.lower() == 'stft':

            Zxx = rfft(frames, n=nfft, axis=0)
            Pxx = np.abs(Zxx)

            unit_test = False
            if unit_test:
                # compute stft using scipy
                _f, _t, _Zxx = signal.stft(data, synthetic_fs, window=window_type,
                                           nperseg=nperseg, noverlap=noverlap,
                                           boundary=None, padded=False, nfft=nfft)

                win_fun = scipy.signal.get_window(window_type, nperseg)
                scale = np.sqrt(1.0 / win_fun.sum() ** 2)
                _Zxx = _Zxx / scale

                assert np.allclose(_Zxx, Zxx)
                assert np.allclose(_f, f)
                assert np.allclose(_t, t)
            return self.get_max_frequencies(f, Pxx, threshold)
        elif self.method.lower() == 'welch':
            raise ValueError('Not implemented')
        elif self.method == 'ESPRIT':
            raise ValueError('Not implemented')
        elif self.method.lower() == 'music':
            raise ValueError('Not implemented')



    def get_max_frequencies(self, f, Pxx, threshold):

        bin_size = f[1] - f[0]
        max_freqs = []
        # for spectrum in np.abs(Zxx.transpose()):  # Transpose to iterate on time frames
        for spectrum in Pxx.transpose():  # Transpose to iterate on time frames
            max_amp = np.amax(spectrum)
            max_freq_idx = np.where(spectrum == max_amp)[0][0]
            # max_freq = spectrum[max_freq_idx]
            # max_freq = quadratic_interpolation(spectrum, max_freq_idx, bin_size)

            max_freq = quadratic_interpolation(data=spectrum,
                                               max_idx=max_freq_idx,
                                               bin_size=bin_size,
                                               freq_band_size=self.freq_band_size,
                                               enf_harmonic_n=self.harmonic_n,
                                               threshold=threshold)

            max_freqs.append(max_freq)

        return max_freqs


    def butter_multi_bandpass_filter_sum(self, data, synthetic_fs, n_harmonics, freq_band_size, order=10):
        from scipy.signal import butter, sosfreqz, sosfilt

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            sos = butter(order, [low, high], analog=False, btype='band', output='sos')
            return sos

        cut_offs = []
        for n in range(1, n_harmonics + 1):  # Start from the second harmonic
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
        return np.mean([sosfilt(sos, data) for sos in sos_list], axis=0)

    def fit_enf(self, data):
        """Extracts a series of ENF values from `data`, one per second.

        :param data: list of signal sample amplitudes
        :param synthetic_fs: the sample rate
        :param nominal_freq: the nominal ENF (in Hz) to look near
        :param freq_band_size: the size of the band around the nominal value in which to look for the ENF
        :param harmonic_n: the harmonic number to look for
        :returns: a list of ENF values, one per second
        """
        # 1. resample
        downsampled_data = ssg.resample_poly(data, self.downsampled_fs, self.synthetic_fs)

        # 2. band pass
        if self.multi_bandpass_kaiser:
            nominal_frequencies = [(1, 100.)]

            cut_off = []
            for n, n_fr in nominal_frequencies:
                cut_off.extend([(n_fr - n * self.freq_band_size),
                                (n_fr + n * self.freq_band_size)])

            taps_kaiser = bandpass_kaiser(cut_off, self.downsampled_fs,
                                          transition_width_hz=1,
                                          ripple_db=70.0)

            b, a = taps_kaiser, [1]
            filtered_data = filtfilt(b=b, a=1, axis=0, x=downsampled_data,
                                     padtype='odd',
                                     padlen=3 * (max(len(b), len(a)) - 1))

        elif self.single_bandpass:
            locut = self.harmonic_n * (self.nominal_freq - self.freq_band_size)
            hicut = self.harmonic_n * (self.nominal_freq + self.freq_band_size)

            filtered_data = self.butter_bandpass_filter(downsampled_data, locut, hicut, self.downsampled_fs, order=10)

        elif self.multi_bandpass_butter:
            filtered_data = self.butter_multi_bandpass_filter_sum(downsampled_data, self.downsampled_fs,
                                                                  n_harmonics=self.harmonic_n,
                                                                  freq_band_size=self.freq_band_size,
                                                                  order=10)
        else:
            filtered_data = downsampled_data

        if self.fm is not None:
            raise ValueError('Not included in this implementation')

        # 3. normalize
        if self.normalize_signals:
            filtered_data_min = np.min(filtered_data)
            filtered_data_max = np.max(filtered_data)
            filtered_data = 2 * (filtered_data - filtered_data_min) / (filtered_data_max - filtered_data_min) - 1

        max_freqs = self.spectrum_estimation(filtered_data, self.downsampled_fs,
                                             self.window_size_seconds,
                                             window_type='hann')
        if self.fm is not None:
            indices = np.arange(1, len(max_freqs) + 1)
            fm_estimation = np.interp(np.linspace(start=1, stop=len(max_freqs),
                                                  num=len(filtered_data)),
                                      indices, max_freqs)

        else:
            fm_estimation = None

        return {
            'downsample': {
                'new_fs': self.downsampled_fs,
            },
            "fm_estimation": fm_estimation,
            "filtered_data": filtered_data,
            'enf': [f / float(self.harmonic_n) for f in max_freqs]
        }
