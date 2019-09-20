import os
import glob
import numpy as np


def list_length(train_lst):
    with open(train_lst) as f:
        for i, l in enumerate(f):
            pass
        return i + 1


def compute_sequence_length(data_root):
    data_lst = glob.glob(os.path.join(data_root, '*', '*.npz'))

    max_seq_len = 0
    for data_path in data_lst:
        d = np.load(data_path)['arr_0']

        if d.shape[0] > max_seq_len:
            print('Changing max_seq_len {}'.format(d.shape[0]))
            max_seq_len = d.shape[0]

    return max_seq_len


if __name__ == '__main__':
    max_seq_length = compute_sequence_length('/dark/Databases/BosphorusSignV2_final/features_mc3_18_avgpool_overlap/')
    print('Done, max_seq_len {}'.format(max_seq_length), flush=True)