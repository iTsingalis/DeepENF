import torch
from torch import Tensor
import torch.nn as nn


def conv_output_size(in_size, kernel, stride=1, padding=0):
    return (in_size - kernel + 2 * padding) // stride + 1


def conv_trans_output_size(in_size, kernel_size, stride=1, padding=0, output_padding=1, dilation=0):
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


# definition of the leaky hard tanh function
def scaled_tanh(input, min_val=-1., max_val=1., trans_val=0.):
    # Step 1: Normalize to [-1, 1] using tanh activation
    normalized_output = torch.tanh(input - trans_val)

    # Step 2: Scale and shift to [min_val, max_val]
    scaled_output = 0.5 * (normalized_output + 1) * (max_val - min_val) + min_val

    # # Step 1: Normalize to [0, 1] using sigmoid activation
    # normalized_output = torch.sigmoid(input)
    #
    # # Step 2: Scale to [min_val, max_val]
    # scaled_output = min_val + (max_val - min_val) * normalized_output

    return scaled_output


# create a class wrapper from PyTorch nn.Module
class Scaledtanh(nn.Module):

    def __init__(self, min_val=-1., max_val=1., trans_val=0.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.trans_val = trans_val

    def forward(self, input: Tensor) -> Tensor:
        return scaled_tanh(input=input, min_val=self.min_val, max_val=self.max_val, trans_val=self.trans_val)


class ENFEstimationModule(nn.Module):

    def __init__(self,
                 signal_dim,
                 n_filters,
                 inner_dim=100,
                 n_layers_linear=2,
                 n_layers_cnn=3,
                 kernel_sizes=None,
                 conv_bias=False,
                 linear_bias=False,
                 freq_band_size=0.1,
                 window_size=1,
                 n_harmonics=1):
        super().__init__()
        self.n_layers_linear = n_layers_linear
        self.n_layers_cnn = n_layers_cnn
        self.n_filters = n_filters
        self.in_features = signal_dim
        self.linear_bias = linear_bias
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.freq_band_size = freq_band_size
        self.window_size = window_size
        self.n_harmonics = n_harmonics

        if len(self.kernel_sizes) == 1:
            self.kernel_sizes = n_layers_cnn * kernel_sizes

        if len(self.n_filters) == 1:
            self.n_filters = n_layers_cnn * n_filters

        assert len(self.kernel_sizes) == n_layers_cnn, 'number of kernels should be equal to the number to conv layers'

        self.in_layer = nn.Sequential(
            nn.Linear(signal_dim, inner_dim, bias=False),
        )

        conv_stride = 1
        max_pool_kernel_size = 2
        max_pool_kernel_size_stride = 2

        conv_layers_out_dim = inner_dim
        mod_cnn = []
        for n in range(self.n_layers_cnn):
            in_filters = self.n_filters[n] if n > 0 else 1

            kernel_size = self.kernel_sizes[n]
            conv_padding = (kernel_size - 1) // 2

            mod_cnn += [
                nn.Conv1d(in_channels=in_filters, out_channels=self.n_filters[n], kernel_size=kernel_size,
                          stride=conv_stride, padding=conv_padding, bias=self.conv_bias),
                nn.BatchNorm1d(self.n_filters[n]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=max_pool_kernel_size_stride),
            ]
            conv_layers_out_dim = conv_output_size(conv_layers_out_dim, kernel_size,
                                                   stride=conv_stride,
                                                   padding=conv_padding)
            print(f'Layer: {n} -- conv_out_dim {conv_layers_out_dim}')
            conv_layers_out_dim = int((conv_layers_out_dim - max_pool_kernel_size) / max_pool_kernel_size_stride) + 1
            print(f'Layer: {n} -- max_pooling_out_dim {conv_layers_out_dim}')

        self.hidden_conv_layers = nn.Sequential(*mod_cnn)

        mod_linear = []
        for n in range(self.n_layers_linear):
            mod_linear += [
                nn.Linear(conv_layers_out_dim * self.n_filters[-1],
                          conv_layers_out_dim * self.n_filters[-1],
                          bias=self.linear_bias),
                nn.BatchNorm1d(conv_layers_out_dim * self.n_filters[-1]),
            ]

        self.linear_layers = nn.Sequential(*mod_linear)

        nominal_freq = 50
        out_layers = []
        for n_harmonic in range(1, self.n_harmonics + 1):  # Work with 100 Hz and 200 Hz
            target_locut = (n_harmonic + 1) * (nominal_freq - self.freq_band_size)
            target_hicut = (n_harmonic + 1) * (nominal_freq + self.freq_band_size)
            out_layers += [
                nn.Sequential(
                    nn.Linear(conv_layers_out_dim * self.n_filters[-1],
                              out_features=self.window_size, bias=self.linear_bias),
                    Scaledtanh(min_val=target_locut, max_val=target_hicut, trans_val=0),
                )
            ]
        self.out_layers = nn.ModuleList(out_layers)

    def forward(self, inp):

        bsz = inp.size(0)
        inp = inp.view(bsz, -1)

        x = self.in_layer(inp).view(bsz, 1, -1)

        x = self.hidden_conv_layers(x).view(bsz, -1)

        # intermediate linear layers
        x = self.linear_layers(x)

        outputs = []
        for n_harmonic in range(self.n_harmonics):
            outputs.append(self.out_layers[n_harmonic](x))

        return outputs


def set_estimation_module(args, signal_dim):
    """
    Create a spectrum representation (sr) module
    """
    if args.enf_est_module_type == 'linear_out':
        net = ENFEstimationModule(signal_dim=signal_dim,
                                  n_filters=args.enf_est_n_filters,
                                  inner_dim=args.enf_est_inner_dim,
                                  n_layers_linear=args.enf_est_n_layers_linear,
                                  n_layers_cnn=args.enf_est_n_layers_cnn,
                                  kernel_sizes=args.enf_est_kernel_sizes,
                                  conv_bias=args.conv_bias,
                                  linear_bias=args.linear_bias,
                                  freq_band_size=args.freq_band_size,
                                  window_size=args.window_size,
                                  n_harmonics=args.n_harmonics_nn
                                  )

    if args.use_cuda:
        net.cuda()

    return net
