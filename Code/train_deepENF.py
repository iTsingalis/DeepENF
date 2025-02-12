"""
Notes:
    https://ai.stackexchange.com/questions/20059/cnn-high-variance-across-multiple-trained-models-what-does-it-mean
"""

import os
import sys

# sys.path.append('/media/blue/tsingalis/DeepENF/')
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from Code.dataset import get_loaders
from Code.modules import set_estimation_module
from Code.utils import set_optimizer, set_scheduler, model_parameters
from Code.utils import save_model, print_args

import logging

import matplotlib.pyplot as plt

# matplotlib.use('Agg')
# plt.switch_backend('agg')

from Code.timer import Timer

my_timer = Timer()

logger = logging.getLogger(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_enf_representation(args,
                             target_fs,
                             enf_est_module,
                             enf_est_optimizer,
                             enf_est_criterion,
                             lr_scheduler,
                             tr_loader,
                             epoch,
                             tb_writer,
                             augmenter=None,
                             disable_tqdm=False):
    """
        Train the frequency-representation module for one epoch
    """
    epoch_start_time = time.time()
    n_training_batches = len(tr_loader)

    enf_est_module.train()
    tr_loss_per_epoch = 0

    for batch_idx, train_data in enumerate(tqdm(tr_loader,
                                                desc='Train loader..',
                                                disable=disable_tqdm)):

        tr_target_frames, tr_reference_frames, tr_reference_frames_freq = train_data

        if augmenter is not None:
            # The tensor has a shape like (tr_batch_size, num_channels, num_samples) and mode is per_example
            tr_target_frames = augmenter(
                samples=tr_target_frames.reshape(tr_target_frames.shape[0], 1, -1),
                sample_rate=target_fs).reshape(tr_target_frames.shape[0], -1)

        if args.use_cuda:
            # tr_reference_frames = tr_reference_frames.cuda()
            tr_reference_frames_freq = [_tr_reference_frames_freq.cuda() for
                                        _tr_reference_frames_freq in tr_reference_frames_freq]
            tr_target_frames = tr_target_frames.cuda()

        enf_est_optimizer.zero_grad()

        tr_output_refrence_frames = enf_est_module(tr_target_frames)

        tr_loss_per_batch = torch.tensor(0.0, device=device)

        for n_harmonic, n_harmonics in enumerate(zip(tr_output_refrence_frames, tr_reference_frames_freq), start=1):
            n_harmonic_out, n_harmonic_ref = n_harmonics
            tr_loss_per_batch += enf_est_criterion(n_harmonic_out / (n_harmonic + 1), n_harmonic_ref / n_harmonic)

        tr_loss_per_batch /= args.n_harmonics_nn

        tr_loss_per_batch.backward(create_graph=False)
        enf_est_optimizer.step()  # Do not comment accidentally !!!

        tr_loss_per_epoch += tr_loss_per_batch.data.item()

        if args.lr_scheduler is not None and args.lr_scheduler in ['CyclicLR']:
            lr_scheduler.step()

    tr_loss_per_epoch /= n_training_batches
    tb_writer.add_scalar(f'loss', tr_loss_per_epoch, epoch)

    if args.lr_scheduler is not None and args.lr_scheduler in ['MultiStepLR', 'ExponentialLR']:
        lr_scheduler.step()

    logger.info("Epochs: %d / %d, "
                "Time: %.1f, "
                "Training loss %.10f, ",
                epoch,
                args.n_epochs_enf_est,
                time.time() - epoch_start_time,
                tr_loss_per_epoch)


def quadratic_interpolation(data, max_idx, bin_size):
    """
        https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    """
    left = data[max_idx - 1]
    center = data[max_idx]
    right = data[max_idx + 1]

    p = 0.5 * (left - right) / (left - 2 * center + right + 1e-4)
    interpolated = (max_idx + p) * bin_size

    return interpolated


def max_frequencies(Zxx, reference_fs, nfft):
    bin_size = reference_fs / nfft

    max_freqs = []
    for i, spectrum in enumerate(np.abs(Zxx.transpose())):  # Transpose to iterate on time frames
        max_amp = np.amax(spectrum)
        max_freq_idx = np.where(spectrum == max_amp)[0][0]
        max_freq = quadratic_interpolation(spectrum, max_freq_idx, bin_size)
        max_freqs.append(max_freq)

    return np.asarray(max_freqs)


def eval_enf_representation(args,
                            enf_est_module,
                            enf_est_criterion,
                            enf_est_optimizer,
                            lr_scheduler,
                            val_loader,
                            epoch,
                            tb_writer,
                            early_stopper=None,
                            disable_tqdm=True,
                            save_best_model=None):
    enf_est_module.eval()
    val_loss_per_epoch = 0

    n_validation_batches = len(val_loader)
    for batch_idx, validation_data in enumerate(tqdm(val_loader,
                                                     desc='Validation loader..',
                                                     disable=disable_tqdm)):

        val_target_signal_frame, val_reference_frames, val_reference_frames_freq = validation_data

        if args.use_cuda:
            val_reference_frames_freq = [_val_reference_frames_freq.cuda() for
                                         _val_reference_frames_freq in val_reference_frames_freq]
            val_target_signal_frame = val_target_signal_frame.cuda()

        with torch.no_grad():
            val_output_target_frames = enf_est_module(val_target_signal_frame)

        val_loss_per_batch = torch.tensor(0.0, device=device)

        for n_harmonic, n_harmonics in enumerate(zip(val_output_target_frames, val_reference_frames_freq), start=1):
            n_harmonic_out, n_harmonic_ref = n_harmonics
            val_loss_per_batch += enf_est_criterion(n_harmonic_out / (n_harmonic + 1), n_harmonic_ref / n_harmonic)

        val_loss_per_batch /= args.n_harmonics_nn

        val_loss_per_epoch += val_loss_per_batch.data.item()

    val_loss_per_epoch /= n_validation_batches

    if args.lr_scheduler is not None and args.lr_scheduler in ['ReduceLROnPlateau']:
        lr_scheduler.step(val_loss_per_epoch)

    tb_writer.add_scalar(f'loss', val_loss_per_epoch, epoch)

    # mean_base = 10 ** -5
    # std_base = 10 ** -4
    '{mean(mse_alg_ordered[alg]) / mean_base: .0f}'
    logger.info(
        "Epochs: %d / %d, "
        "Validation loss %.10f, ",
        # "Validation FFT mean MSE %.0f x 10^{{-5}}, "
        # "Validation FFT std MSE %.0f x 10^{{-4}}, ",
        epoch,
        args.n_epochs_enf_est,
        val_loss_per_epoch,
        # mean_mse / mean_base,
        # stdev_mse / std_base
    )

    if save_best_model is not None:
        save_best_model(val_loss_per_epoch, epoch, enf_est_module, enf_est_optimizer,
                        lr_scheduler, enf_est_criterion,
                        'best_model.pth')

    # np.random.seed(2020)
    # rand_row = np.random.choice(val_reference_frames.shape[0], 1, replace=False)
    # compute_stats(rand_row, val_reference_frames, val_output_target_frames,
    #               reference_fs, moving_average_window_size, epoch, train_or_val='val')

    if early_stopper is not None and early_stopper.early_stop(val_loss_per_epoch):
        raise ValueError('Early stop in validation.')


class EarlyStopper:
    """
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """

    def __init__(self, patience=1, min_delta=1e-1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (validation_loss > (self.min_validation_loss + self.min_delta) or
              abs(validation_loss - self.min_validation_loss) < self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def check_run_folder(exp_folder):
    run = 1
    run_folder = os.path.join(exp_folder, 'run{}'.format(run))

    if not os.path.exists(run_folder):
        Path(run_folder).mkdir(parents=True, exist_ok=True)

        print("Path {} created".format(run_folder))
        return run_folder

    while os.path.exists(run_folder):
        run += 1
        run_folder = os.path.join(exp_folder, 'run{}'.format(run))
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    print("Path {} created".format(run_folder))
    return run_folder


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class ConcordanceCC(nn.Module):
    def __init__(self):
        super(ConcordanceCC, self).__init__()

    def concordance_cc(self, inputs, targets):
        """Defines concordance loss for training the model.

        Args:
           inputs: prediction of the model (PyTorch tensor).
           ground_truth: ground truth values (PyTorch tensor).
        Returns:
           The concordance value.
        """
        pred_mean = torch.mean(inputs, dim=0)
        pred_var = torch.var(inputs, dim=0)
        gt_mean = torch.mean(targets, dim=0)
        gt_var = torch.var(targets, dim=0)

        mean_cent_prod = torch.mean((inputs - pred_mean) * (targets - gt_mean))

        return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + (pred_mean - gt_mean) ** 2)

    def forward(self, inputs, targets):
        return self.concordance_cc(inputs, targets)


def main():

    """
--enf_est_criterion
L1Loss
--n_epochs_enf_est
250
--augment
--min_snr_in_db
-2
--max_snr_in_db
2
--enf_est_n_filters
16
--enf_est_inner_dim
500
--enf_est_n_layers_linear
5
--enf_est_n_layers_cnn
5
--fold
0
--numpy_seed
100
--n_folds
5
--enf_est_kernel_sizes
3
--downsample_fs
800
--lr_estimation
5e-4
--tr_batch_size
256
--val_batch_size
512
--lr_scheduler
ReduceLROnPlateau
--enf_est_module_type
linear_out
--nfft_scale
0
--kaiser_bandpass
--freq_band_size
0.1
--window_size
1
--n_harmonics_nn
1
--n_harmonics_data
1
--ripple_db
15
--transition_width_hz
0.1
"""
    train_tb_writer = SummaryWriter(os.path.join(run_dir, 'logs', 'train'))
    val_tb_writer = SummaryWriter(os.path.join(run_dir, 'logs', 'val'))

    file_handler = logging.FileHandler(filename=os.path.join(run_dir, 'logs', 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    if args.augment:
        p = .4
        mode = p_mode = "per_example"
        from torch_audiomentations import Compose, AddColoredNoise, Shift, PitchShift, PolarityInversion, BandPassFilter
        augmenter = Compose([
            PolarityInversion(p=p, mode=mode, p_mode=p_mode),
            Shift(min_shift=-0.1, max_shift=0.1, p=p, mode=mode, p_mode=p_mode),
            AddColoredNoise(p=p, min_snr_in_db=args.min_snr_in_db,
                            max_snr_in_db=args.max_snr_in_db, mode=mode, p_mode=p_mode)
        ])
        with open(os.path.join(run_dir, 'augmenter.txt'), "w") as text_file:
            print(augmenter, file=text_file)
    else:
        augmenter = None

    if args.hua_rfa:
        target_folder = f"{root_folder}/Data/Hua/H1_rfa/"
    else:
        target_folder = f"{root_folder}/Data/Hua/H1/"

    ref_folder = f"{root_folder}/Data/Hua/H1_ref/"

    validation_split = .5
    val_n_samples = int(validation_split * args.tr_n_samples) if args.tr_n_samples is not None else None

    loaders, signals_dim, signals_fs = get_loaders(args, target_folder, ref_folder,
                                                   tr_n_samples=args.tr_n_samples,
                                                   val_n_samples=val_n_samples,
                                                   tst_n_samples=val_n_samples,
                                                   folds_dir=f'{root_folder}/Data/Folds/')
    tr_loader, val_loader, tst_loader = loaders
    target_signal_dim, reference_frame_dim = signals_dim

    target_fs, reference_fs = signals_fs

    enf_est_module = set_estimation_module(args, target_signal_dim)
    with open(os.path.join(run_dir, 'network_arch.txt'), "w") as text_file:
        print(enf_est_module, file=text_file)
    print(enf_est_module)

    enf_est_optimizer = set_optimizer(args, enf_est_module, module_type='estimation')
    print(enf_est_optimizer)

    with open(os.path.join(run_dir, 'enf_est_optimizer.json'), 'w') as fp:
        json.dump(enf_est_optimizer.state_dict(), fp, indent=4)

    if args.enf_est_criterion == 'SmoothL1Loss':
        enf_est_criterion = torch.nn.SmoothL1Loss()
    elif args.enf_est_criterion == 'MSELoss':
        enf_est_criterion = torch.nn.MSELoss(reduction='mean')
    elif args.enf_est_criterion == 'L1Loss':
        enf_est_criterion = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    elif args.enf_est_criterion == 'HuberLoss':
        enf_est_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.35)
    elif args.enf_est_criterion == 'CosineEmbeddingLoss':
        enf_est_criterion = torch.nn.CosineEmbeddingLoss()
    elif args.enf_est_criterion == 'RMSLELoss':
        enf_est_criterion = RMSLELoss()
    elif args.enf_est_criterion == 'ConcordanceCC':
        enf_est_criterion = ConcordanceCC()
    elif args.enf_est_criterion == 'RMSLELoss':
        enf_est_criterion = RMSLELoss()

    with open(os.path.join(run_dir, 'enf_est_criterion.json'), 'w') as fp:
        json.dump(enf_est_criterion.state_dict(), fp, indent=4)

    logger.info(
        '[Network] Number of parameters in the ENF-estimation module : %.3f M' % (
                model_parameters(enf_est_module) / 1e6))

    if args.lr_scheduler:
        lr_scheduler = set_scheduler(args, enf_est_optimizer, tr_loader=tr_loader, lr_tuner=None)
        print(lr_scheduler.state_dict())
        with open(os.path.join(run_dir, 'lr_scheduler.json'), 'w') as fp:
            json.dump(lr_scheduler.state_dict(), fp, indent=4)
    else:
        lr_scheduler = None

    start_epoch = 1
    final_save_model = False
    my_timer.start()
    for epoch in tqdm(range(start_epoch, args.n_epochs_enf_est + 1), disable=True):

        train_enf_representation(args=args,
                                 target_fs=target_fs,
                                 enf_est_module=enf_est_module,
                                 enf_est_optimizer=enf_est_optimizer,
                                 enf_est_criterion=enf_est_criterion,
                                 lr_scheduler=lr_scheduler,
                                 tr_loader=tr_loader,
                                 epoch=epoch,
                                 tb_writer=train_tb_writer,
                                 augmenter=augmenter,
                                 disable_tqdm=False)
        try:
            eval_enf_representation(args=args,
                                    enf_est_module=enf_est_module,
                                    enf_est_criterion=enf_est_criterion,
                                    enf_est_optimizer=enf_est_optimizer,
                                    lr_scheduler=lr_scheduler,
                                    val_loader=val_loader,
                                    epoch=epoch,
                                    tb_writer=val_tb_writer,
                                    disable_tqdm=True)
        except ValueError:
            # final_save_model = True
            pass
            # break
        finally:
            if epoch == args.n_epochs_enf_est or epoch % 5 == 0 or final_save_model:
                # save the trained model weights for a final time
                print('Save final model..')
                save_model(epoch, enf_est_module, enf_est_optimizer, lr_scheduler, tuned_lr=None,
                           args=args, output_dir=run_dir, file_name=f'final_model_e{epoch}.pth')

    my_timer.stop(tag='Training', verbose=True)

    train_tb_writer.close()
    val_tb_writer.close()

    print('That is all folks!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ENF')

    parser.add_argument('--run_dir',
                        type=str, default='./checkpoint/experiment_name',
                        help='output directory')

    # Data
    # parser.add_argument('--audio_length', type=int, default=133623)
    parser.add_argument('--tr_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--tst_batch_size', type=int, default=256)

    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--no_cuda', action='store_true',
                        help="avoid using CUDA when available")
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)
    parser.add_argument('--nfft_scale', type=int, default=0)

    parser.add_argument('--fold', type=int, required=True,
                        choices=[0, 1, 2, 3, 4])

    parser.add_argument('--n_folds',
                        type=int, required=True, default=5,
                        choices=[3, 5, 10])

    parser.add_argument('--lr_estimation', type=float, default=1e-5,
                        help='initial learning rate for adam optimizer used for the estimation-representation module')

    parser.add_argument('--n_epochs_enf_est', type=int, default=250,
                        help='number of epochs used to train the enf representation (fr) module')

    parser.add_argument('--downsample_fs', type=int,
                        default=8000,
                        help='The sampling frequency of target signal.')

    parser.add_argument('--window_size_seconds', type=int, default=16,
                        help='Window size in seconds.')

    parser.add_argument('--enf_est_criterion', type=str, default='MSELoss',
                        choices=["SmoothL1Loss",
                                 "MSELoss",
                                 "L1Loss",
                                 "HuberLoss",
                                 "CosineEmbeddingLoss",
                                 "RMSLELoss",
                                 "ConcordanceCC"],
                        help='The criterion used to train the enf representation (fr) module')

    parser.add_argument('--enf_est_n_layers_cnn', type=int, default=2,
                        help='Number of convolutional layers in the enf representation (sr) module')

    parser.add_argument('--enf_est_n_layers_linear', type=int, default=2,
                        help='Number of linear output layers in the enf representation (sr) module')

    parser.add_argument('--enf_est_n_filters', nargs='+', type=int, default=16,
                        help='Number of filters per layer in the enf representation (sr) module')

    parser.add_argument('--enf_est_inner_dim', type=int, default=200,
                        help='Dimension after first linear transformation')

    parser.add_argument('--enf_est_kernel_sizes', '--list', nargs='+', type=int,
                        help='Filter size in the convolutional blocks of the enf representation (sr) module',
                        required=True)

    parser.add_argument('--min_snr_in_db', type=int, default=-10, help='Min SRN in db')

    parser.add_argument('--max_snr_in_db', type=int, default=10, help='Max SRN in db')

    parser.add_argument('--linear_bias', action='store_true', help="Use bias term in linear layers")

    parser.add_argument('--conv_bias', action='store_true', help="Use bias term in conv layers")

    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=["CyclicLR", "MultiStepLR", "ExponentialLR", "ReduceLROnPlateau"],
                        help='Use lr scheduler')

    parser.add_argument('--augment', action='store_true', help="Use lr scheduler")

    parser.add_argument('--tune_lr', action='store_true', help="Use lr tuner")

    parser.add_argument('--butter_bandpass', action='store_true', help="Use bandpass filtering")

    parser.add_argument('--butter_multi_bandpass', action='store_true', help="Use bandpass filtering")

    parser.add_argument('--kaiser_bandpass', action='store_true', help="Use bandpass filtering")

    parser.add_argument('--enf_est_module_type', type=str, choices=["linear_out", "conv_trans_out"],
                        help='type of the fr module: [linear_out | conv_trans_out]')

    parser.add_argument('--enf_est_upsampling', type=int, default=8,
                        help='stride of the transposed convolution, upsampling * inner_dim = enf_est_size')

    parser.add_argument('--window_size', type=int, default=1,
                        help='The number of the frequencies in the window to keep.')

    parser.add_argument('--freq_band_size', type=float, default=0.1, required=True,
                        help='The frequency band for the bandpass filter')

    parser.add_argument('--transition_width_hz', type=float, default=0.1, required=False,
                        help='The transition width in Hz for the bandpass filter')

    parser.add_argument('--ripple_db', type=float, default=15, required=False,
                        help='The riple in Db for the bandpass filter')

    parser.add_argument('--n_harmonics_data', type=int, default=1,
                        required=False, choices=[1, 2, 3, 4, 5],
                        help='The number of harmonics to filter data')

    parser.add_argument('--n_harmonics_nn', type=int, default=1,
                        required=False, choices=[1, 2, 3, 4, 5],
                        help='The number of harmonics to use in the estimation')

    parser.add_argument('--hua_rfa', action='store_true', help="Use RFA data.")

    parser.add_argument('--tr_n_samples', type=int, default=None, help='NUmber of training samples')


    args = parser.parse_args()

    assert args.window_size % 2 != 0, 'window length must be odd'

    assert not (args.butter_bandpass and args.kaiser_bandpass and args.butter_multi_bandpass), \
        'Only butter_bandpass or kaiser_bandpass can be enabled'

    assert args.n_harmonics_nn <= args.n_harmonics_data, 'Note n_harmonics_nn <= n_harmonics_data'

    if args.butter_bandpass:
        assert args.n_harmonics_data == 1, 'butter_bandpass works only with a single harmonic'

    # print_args(logger, args)

    args_dict = vars(args)
    print(args_dict)

    #Change this according to your folders
    root_folder = '/media/blue/tsingalis/DeepENFv2/DeepENF'
    # root_folder = '/media/blue/tsingalis/gitRepositories/DeepENF'

    output_dir = f'{root_folder}/experiments_temp'
    run_dir = check_run_folder(output_dir)

    Path(os.path.join(run_dir, 'output')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(run_dir, 'rfft')).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(run_dir, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp, indent=4)

    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_args(args, logger)

    main()
