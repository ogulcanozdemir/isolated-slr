from datasets.dataset import GenericDataset
from utils.constants import SamplingType
from utils.logger import logger


import torch


class DatasetLSTM(GenericDataset):

    def __init__(self,
                 root_dir,
                 dataset='',
                 test_mode=False,
                 split=None,
                 modality=None,
                 sampling=None,
                 transform=None,
                 sequence_len=None,
                 feature_extract=False):
        super(DatasetLSTM, self).__init__(root_dir=root_dir,
                                          dataset=dataset,
                                          split=split,
                                          modality=modality,
                                          sampling=sampling,
                                          transform=transform,
                                          test_mode=test_mode,
                                          feature_extract=feature_extract)

        self.sequence_length = sequence_len

        self.print_summary()

    def print_summary(self):
        logger.info('Initializing dataset: {}, split {}'.format(self.dataset, self.split.value))
        logger.info('Dataset path: {}'.format(self.root_dir))
        logger.info('Number of classes: {}'.format(self.num_classes))
        logger.info('')
        logger.info('Input modality: {}'.format(self.modality))
        logger.info('Input sampling: {}'.format(self.sampling))
        # logger.info('Max sequence length: {}'.format(self.sequence_length))
        logger.info('Input transforms: ')
        for t in self.transform.transforms:
            logger.info('\t' + type(t).__name__ + ': ' + str(vars(t)))
        logger.info('==========================================')

    def __getitem__(self, index: int):
        if not self.feature_extract:
            sequence = self.load_sequence(self.videos[index])
            label = self.labels[index]

            sequence = self.transform(sequence)

            if self.test_mode:
                return sequence, label, self.videos[index]
            else:
                return sequence, label
        else:
            raise NotImplementedError

    def get_segment_indices(self, sequence):
        seq_len = sequence.shape[0]

        segment_indices = []
        if self.sampling == SamplingType.ALL.value:
            # return all of frames
            segment_indices = list(range(0, seq_len))

        elif self.sampling == SamplingType.KEYFRAME.value:
            # return keyframes
            raise NotImplementedError

        return segment_indices


class PadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_collate(self, batch):
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        xs = []
        ys = []
        for b in batch:
            x = b[0]
            y = b[1]

            xs.append(pad_tensor(x, pad=max_len, dim=self.dim))
            ys.append(y)

        # batch = map(lambda x, y: (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        # xs = torch.stack(map(lambda x: x[0], batch), dim=self.dim)
        # ys = torch.LongTensor(map(lambda x: x[1], batch))

        xs = torch.stack(xs, dim=self.dim)
        ys = torch.LongTensor(ys)

        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


def pad_tensor(vec, pad, dim=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
