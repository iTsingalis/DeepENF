from DeepENF.Code.dataset import target_ref_pair
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from pathlib import Path

import os
import ast

import numpy as np


def get_pair_split():
    data_pair = target_ref_pair(target_folder, ref_folder)

    dataset_size = len(data_pair)
    indices = list(range(dataset_size))

    np.random.seed(100)
    np.random.shuffle(indices)
    data_pair = [data_pair[k] for k in indices]

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)

    fold_folder = os.path.join(output_dir, f'folds{n_splits}')
    Path(fold_folder).mkdir(parents=True, exist_ok=True)

    kf_pairs = list(kf.split(data_pair))
    for i, (train_index, test_index) in enumerate(kf_pairs):
        _train_index, _val_index = train_test_split(train_index, train_size=0.9)
        tr_data_pair = [data_pair[j] for j in _train_index]
        val_data_pair = [data_pair[j] for j in _val_index]
        tst_data_pair = [data_pair[j] for j in test_index]

        save = False
        if save:
            with open(os.path.join(fold_folder, f"tr_data_pair_fold_{i}.txt"), 'w') as f:
                f.write(str(tr_data_pair))

            with open(os.path.join(fold_folder, f"val_data_pair_fold_{i}.txt"), 'w') as f:
                f.write(str(val_data_pair))

            with open(os.path.join(fold_folder, f"tst_data_pair_fold_{i}.txt"), 'w') as f:
                f.write(str(tst_data_pair))


def validate_data_split(n_splits):
    data_pair = target_ref_pair(target_folder, ref_folder)

    dataset_size = len(data_pair)
    indices = list(range(dataset_size))

    np.random.seed(100)
    np.random.shuffle(indices)
    data_pair = [data_pair[k] for k in indices]
    org_wav_test_data = [td[0] for td in data_pair]
    org_ref_test_data = [td[1] for td in data_pair]

    fold_folder = os.path.join(output_dir, f'folds{n_splits}')

    wav_test_data, ref_test_data = [], []
    for i in range(n_splits):
        with open(os.path.join(fold_folder, f"tr_data_pair_fold_{i}.txt"), 'r') as f:
            retrieved_tr_data_pair = ast.literal_eval(f.read())

        with open(os.path.join(fold_folder, f"val_data_pair_fold_{i}.txt"), 'r') as f:
            retrieved_val_data_pair = ast.literal_eval(f.read())

        with open(os.path.join(fold_folder, f"tst_data_pair_fold_{i}.txt"), 'r') as f:
            retrieved_tst_data_pair = ast.literal_eval(f.read())

        wav_test_data.append([td[0] for td in retrieved_tst_data_pair])
        ref_test_data.append([td[1] for td in retrieved_tst_data_pair])

    wav_test_data = np.hstack(wav_test_data)
    ref_test_data = np.hstack(ref_test_data)

    assert sorted(wav_test_data) == sorted(org_wav_test_data)
    assert sorted(ref_test_data) == sorted(org_ref_test_data)

    print()


if __name__ == "__main__":
    output_dir = '/media/blue/tsingalis/DeepENF/outputs/'
    target_folder = "/media/blue/tsingalis/DeepENF/Data/Hua/H1/"
    ref_folder = "/media/blue/tsingalis/DeepENF/Data/Hua/H1_ref/"

    validate_data_split(n_splits=5)
    # get_pair_split()
