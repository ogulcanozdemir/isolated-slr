import pandas as pd
import os


def prepare_toydata(csv_data_file, csv_class_file, split_path, test_user):
    # prepare class indices
    class_csv = pd.read_csv(csv_class_file, delimiter=',', dtype=str)
    print(class_csv)

    label2idx = {}
    with open(os.path.join(split_path, 'class_indices.txt'), 'w') as f:
        class_csv_filtered = class_csv[['ClassID', 'ClassName_tr']]
        for idx, (class_id, class_name) in enumerate(class_csv_filtered.values):
            f.write(str(idx) + ' ' + class_id + ' ' + class_name + '\n')
            label2idx[class_id] = idx
    # prepare train and test splits

    data_csv = pd.read_csv(csv_data_file, delimiter=',', dtype=str)
    print(data_csv)

    with open(os.path.join(split_path, 'train.txt'), 'w') as f:
        data_csv_filtered_train = data_csv.loc[data_csv.UserID != 'User_'+test_user][['ClassID', 'UserID', 'RepeatID']]
        for class_id, user_id, repeat_id in data_csv_filtered_train.values:
            f.write(os.path.join(class_id, user_id + '_{:03d}'.format(int(repeat_id))) + ' ' + str(label2idx[class_id]) + '\n')

    with open(os.path.join(split_path, 'test.txt'), 'w') as f:
        data_csv_filtered_test = data_csv.loc[data_csv.UserID == 'User_'+test_user][['ClassID', 'UserID', 'RepeatID']]
        for class_id, user_id, repeat_id in data_csv_filtered_test.values:
            f.write(os.path.join(class_id, user_id + '_{:03d}'.format(int(repeat_id))) + ' ' + str(label2idx[class_id]) + '\n')


def prepare_cslr(data_path, split_path):
    if not os.path.exists(split_path):
        os.makedirs(split_path)

    train_ids = list(range(1, 37))
    test_ids = list(range(37, 51))

    class_dir = sorted(os.listdir(data_path))
    print(class_dir)

    label2idx = {}
    with open(os.path.join(split_path, 'class_indices.txt'), 'w') as f:
        for idx, class_id in enumerate(class_dir):
            f.write(str(idx) + ' ' + class_id + ' ' + class_id + '\n')
            label2idx[class_id] = idx

    with open(os.path.join(split_path, 'train.txt'), 'w') as f:
        for class_id in class_dir:
            class_user_dir = sorted(os.listdir(os.path.join(data_path, class_id)))
            for class_user in class_user_dir:
                user_id = int(class_user.split('_')[0][1:])
                if user_id in train_ids:
                    f.write(os.path.join(class_id, class_user) + ' ' + str(label2idx[class_id]) + '\n')

    with open(os.path.join(split_path, 'test.txt'), 'w') as f:
        for class_id in class_dir:
            class_user_dir = sorted(os.listdir(os.path.join(data_path, class_id)))
            for class_user in class_user_dir:
                user_id = int(class_user.split('_')[0][1:])
                if user_id in test_ids:
                    f.write(os.path.join(class_id, class_user) + ' ' + str(label2idx[class_id]) + '\n')


if __name__ == '__main__':
    # data_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\frames_centered_360x360_180x0'
    # csv_data_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\BosphorusSignV2_Toydata.csv'
    # csv_class_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\BosphorusSignV2_Toydata_classindices.csv'
    # csv_data_path = '/dark/Databases/BosphorusSignV2_final/BosphorusSignV2.csv'
    # csv_class_path = '/dark/Databases/BosphorusSignV2_final/BosphorusSignV2_classindices.csv'
    #
    # assert os.path.exists(data_path), "Data root does not exists in {}".format(data_path)
    # assert os.path.exists(csv_data_path), "CSV file does not exists in {}".format(csv_data_path)
    # assert os.path.exists(csv_class_path), "CSV file does not exists in {}".format(csv_class_path)
    #
    # prepare_toydata(csv_data_file=csv_data_path,
    #                 csv_class_file=csv_class_path,
    #                 split_path=os.path.join(os.getcwd(), 'splits', 'bsign_7'),
    #                 test_user='7')

    prepare_cslr(data_path='/dark/Databases/CSLR/isolated_slr500_rgb',
                    split_path=os.path.join(os.getcwd(), 'splits', 'cslr'))

    print('done')
