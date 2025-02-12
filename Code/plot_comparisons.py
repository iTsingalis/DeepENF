import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev, median
import mpltex
import pathlib
import matplotlib

linestyles = mpltex.linestyle_generator(lines=['-'])


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def plt_mse():
    stat_type = 'MSE'

    tested_algs = ['Hua', 'DeepENF']
    # Change this according to your needs
    exp_path = '/media/blue/tsingalis/gitRepositories/DeepENF/experiments/RFA/'
    X = []
    for fold in range(5):
        csv_path = os.path.join(exp_path, f'Fold{fold}', 'enf', '110')
        pd_csv = pd.read_csv(os.path.join(csv_path, f'{stat_type}_tst_single.csv'))
        statics_algs_pd = pd_csv  # .set_index('wav_files').reset_index()
        statics_algs_pd['fold'] = f'fold{fold}'
        X.append(statics_algs_pd)

    X = pd.concat(X, ignore_index=True)
    X['abs_diff'] = abs(X['Hua'] - X['DeepENF'])

    X.sort_values(by=['abs_diff'], ascending=False, inplace=True)
    keep_more_noisy = False
    if keep_more_noisy:
        X = X[X['abs_diff'] > 5e-5]

    print(X[4:10][['fold', 'wav_files', 'Hua', 'DeepENF']])
    print(X[4:10][['Hua', 'DeepENF']].mean())

    import seaborn as sns
    fig, ax = plt.subplots()
    df = X[['Hua', 'DeepENF']]
    bplot = df.boxplot(['Hua', 'DeepENF'], figsize=(10, 10), showfliers=True)
    bplot.set_xticks([1, 2], ['P-MLE', 'E-DeepENF'])
    bplot.set_ylabel('MSE', size=15)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-5, -5))

    plt.xticks(size=15)
    plt.yticks(size=15)

    plt.savefig(os.path.join('./', f'{stat_type}_single_hua_box_plot.png'), bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join('./', f'{stat_type}_single_hua_box_plot.eps'),
                bbox_inches='tight', pad_inches=0.1, format='eps')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for alg in tested_algs:
        if alg == 'Hua':
            ax.plot(range(0, len(X[alg].values), 1), X[alg].values, label='P-MLE', **next(linestyles))
        else:
            ax.plot(range(0, len(X[alg].values), 1), X[alg].values, label='E-' + alg, **next(linestyles))

        ax.set_xticklabels([])

    plt.xlabel('Samples in the WHU dataset')
    plt.ylabel(stat_type)
    plt.yticks(rotation=50)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # plt.yscale('log', base=10)

    plt.xticks(range(0, len(X), 10))

    # # ax.set_xticklabels([f'{i}'.zfill(3) + '.wav' for i in X.index],
    # #                    rotation='vertical')
    plt.xticks(range(0, len(X[alg].values)),
               [f'{i}'.zfill(3) + '.wav' for i in X.index],
               rotation=90, fontsize=8)

    ax.yaxis.set_major_formatter(OOMFormatter(-5, "%1.1f"))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-5, -5))
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    # plt.savefig(os.path.join('./', f'{stat_type}_single_hua.png'), bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(os.path.join('./', f'{stat_type}_single_hua.eps'),
    #             bbox_inches='tight', pad_inches=0.1, format='eps')

    # plt.close(fig)

    mse_base = 10 ** -5
    std_base = 10 ** -4

    results = {}
    for alg in tested_algs:

        mean_per_alg_sci = float('{:.4f}'.format(mean(X[alg]) / mse_base))
        std_per_alg_sci = float('{:.4f}'.format(stdev(X[alg]) / std_base))

        print(f'Alg: {alg} -- mean {stat_type}: {mean_per_alg_sci} x 10^{{-5}}'
              f'-- std {stat_type}: {std_per_alg_sci} x 10^{{-4}} -- '
              f'mean {stat_type}: {mean(X[alg])} '
              f'-- std {stat_type}: {stdev(X[alg])} '
              f'median {stat_type}: {median(X[alg])} ')

        results.update({alg: {
            f'std {stat_type} sci': f'{mean_per_alg_sci} x 10^{{-4}}',
            f'mean {stat_type} sci': f'{std_per_alg_sci} x 10^{{-5}}',
            f'std {stat_type}': stdev(X[alg]),
            f'mean {stat_type}': mean(X[alg]),
            f'median {stat_type}': median(X[alg]),
        }})


def check_hua():

    hua_path = '/media/blue/tsingalis/DeepENF/hua_outputs/'
    # hua_alg_pd = pd.read_csv(os.path.join(hua_path,
    #                                       f'stats_wrt_MSE_single.txt'))

    # hua_alg = list(hua_alg_pd.set_index('index_name_order')['MSE_P_MLE_order'].to_dict().items())
    mse_per_wav = []
    from statistics import mean, stdev, median
    for txt_file in pathlib.Path(os.path.join(hua_path, 'f_ref')).glob('*.txt'):
        base_name_txt_file = os.path.basename(txt_file)

        with open(os.path.join(hua_path, 'f_ref', base_name_txt_file), "r") as infile:
            ref_enf = [float(line.rstrip('\n')) for line in infile]

        with open(os.path.join(hua_path, 'f_P_MLE_ENF', base_name_txt_file), "r") as infile:
            target_enf = [float(line.rstrip('\n')) for line in infile]

        mse_per_wav.append(np.linalg.norm(np.array(ref_enf) - np.array(target_enf)) ** 2 / len(ref_enf))

    mse_per_wav = np.array(mse_per_wav)
    print(f'Sample size: {len(mse_per_wav)}')
    q25, q75 = np.percentile(mse_per_wav, [25, 75])
    bin_width = 2 * (q75 - q25) * len(mse_per_wav) ** (-1 / 3)
    bins = round((mse_per_wav.max() - mse_per_wav.min()) / bin_width)
    print("Freedman–Diaconis number of bins:", bins)

    import seaborn as sns
    sns.displot(mse_per_wav, bins=82, kde=True);
    plt.show()

    print(f'Mean MSE: {mean(mse_per_wav)}')
    from decimal import Decimal

    print(mean(mse_per_wav))
    print(float('{:.4f}'.format(mean(mse_per_wav) / 10 ** (-5))))


if __name__ == '__main__':
    plt_mse()
