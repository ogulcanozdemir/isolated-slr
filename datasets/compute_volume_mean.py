import numpy as np
import cv2 as cv
import os


def list_length(train_lst):
    with open(train_lst) as f:
        for i, l in enumerate(f):
            pass
        return i + 1


def compute_volume_mean(data_root, train_lst, num_frames=16, new_w_h_size=112):
    count = 0

    length = list_length(train_lst)
    with open(train_lst) as f:
        sum = np.zeros((new_w_h_size, new_w_h_size, 3))
        for idx, line in enumerate(f):
            print('Reading line {}/{}'.format(idx, length), flush=True)
            vid_path = line.split()[0]

            n_frames = len(os.listdir(os.path.join(data_root, vid_path)))
            for i in range(0, n_frames):
                img = cv.imread(os.path.join(data_root, vid_path, '{:06d}.jpg'.format(i)))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, (new_w_h_size, new_w_h_size))
                sum += img

            count += n_frames

    mean = sum / float(count)
    print(mean)
    return np.repeat(mean[np.newaxis, :, :, :], num_frames, axis=0)


if __name__ == '__main__':
    mean_16 = compute_volume_mean('/dark/Databases/BosphorusSignV2_final/frames_112x112/',
                                  '/raid/users/oozdemir/code/untitled-slr-project/datasets/splits/bsign/train.txt')
    # mean_16 = compute_volume_mean('D:\\Databases\\BosphorusSignV2\\Toydata\\frames_120x120',
    #                                       'D:\\Development\\workspaces\\untitled-slr-project\\datasets\\splits\\toydata\\train.txt')
    np.save('crop_mean_16_toydata_112x112.npy', mean_16)
    print('Done', flush=True)