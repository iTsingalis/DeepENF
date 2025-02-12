# sys.path.append('/media/blue/tsingalis/DeepENF/')
import os
import torch
import errno
import numpy as np
import matplotlib.pyplot as plt
from Code.modules import set_estimation_module


def print_args(logger, args):
    message = ''
    for k, v in sorted(vars(args).items()):
        message += '\n{:>30}: {:<30}'.format(str(k), str(v))
    logger.info(message)

    args_path = os.path.join(args.output_dir, 'run.args')
    with open(args_path, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')


# plt.interactive(False)
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def rolling_avg(data, rolling_window=2):
    average_data = []

    for ind in range(len(data) - rolling_window + 1):
        average_data.append(np.mean(data[ind:ind + rolling_window]))

    for ind in range(rolling_window - 1):
        # average_data.insert(0, np.nan)
        average_data.insert(0, np.nan)

    return average_data


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def get_optim_lr(optimizer):
    return [grp["lr"] for grp in optimizer.param_groups]


def set_optim_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
    print(f'Optimizers lr is tuned to {lr}')


def print_args(args, logger=None):
    message = ''
    for k, v in sorted(vars(args).items()):
        message += '\n{:>30}: {:<30}'.format(str(k), str(v))
    if logger is not None:
        logger.info(message)
    print(message)


def set_scheduler(args, optimizer, tr_loader=None, lr_tuner=None):
    # scheduler = EarlyStopper(patience=3, min_delta=10)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if lr_tuner is None:
        base_lr, max_lr = args.lr_estimation, 5 * args.lr_estimation
    else:
        base_lr, max_lr = lr_tuner['base_lr'], lr_tuner['max_lr']

    if args.lr_scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[30, 100, 200],
                                                            # milestones=[55, 80, 100],
                                                            # milestones=[10, 15, 20],
                                                            # milestones=[70, 100, 150],
                                                            # milestones=[15, 20],
                                                            gamma=0.1,
                                                            verbose=True)
    elif args.lr_scheduler == 'ExponentialLR':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, verbose=True)
    elif args.lr_scheduler == 'CyclicLR':
        step_size_up = int(8 * len(tr_loader.dataset) / args.tr_batch_size) if tr_loader is not None else 2000
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                         step_size_up=step_size_up,
                                                         base_lr=base_lr,
                                                         max_lr=max_lr,
                                                         cycle_momentum=False, verbose=False)
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  'min', patience=7,
                                                                  factor=0.1, verbose=True)

    return lr_scheduler


def set_optimizer(args, module, module_type):
    if module_type == 'estimation':
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_estimation, weight_decay=1e-6)
    elif module_type == 'detection':
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-4, weight_decay=1e-6)
    else:
        raise (ValueError('Expected module_type to be fr or fc but got {}'.format(module_type)))

    return optimizer


def model_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params


def load(model_pth, lr_tuner_pth, module_type, device=torch.device('cpu')):
    checkpoint = torch.load(model_pth, map_location=device)
    args = checkpoint['args']
    if device == torch.device('cpu'):
        args.use_cuda = False
    if module_type == 'estimation':
        model = set_estimation_module(args, checkpoint['in_features_dim'])
    elif module_type == 'detection':
        pass
    else:
        raise NotImplementedError('Module type not recognized')
    model.load_state_dict(checkpoint['model'])

    optimizer = set_optimizer(args, model, module_type)

    if checkpoint["scheduler"] is not None:
        if checkpoint['lr_finder'] is not None:
            scheduler = set_scheduler(args, optimizer, lr_tuner=checkpoint['lr_finder'])
        else:
            scheduler = set_scheduler(args, optimizer, lr_tuner=None)

        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        scheduler = None

    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, scheduler, args, checkpoint['epoch']


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save_model(epoch, model, optimizer, scheduler, tuned_lr, args, output_dir, file_name):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'in_features_dim': model.in_features,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'lr_finder': tuned_lr if tuned_lr is not None else None,
        'args': args,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cp = os.path.join(output_dir, 'last.pth')
    fn = os.path.join(output_dir, file_name)
    torch.save(checkpoint, fn)
    symlink_force(fn, cp)


def plot_stft(f, t, zxx, loclip_f=None, hiclip_f=None, plt_block=True):
    """
    Plots STFT output on the given ax.
    """
    fig, ax = plt.subplots()
    plt.title("Target frequency spectra over time")

    bin_size = f[1] - f[0]
    lindex = int((loclip_f) / bin_size) if loclip_f is not None else 0
    hindex = int((hiclip_f) / bin_size) if hiclip_f is not None else -1

    ax.pcolormesh(t, f[lindex:hindex], np.abs(zxx[lindex:hindex]), shading='gouraud')
    plt.show(block=plt_block)
