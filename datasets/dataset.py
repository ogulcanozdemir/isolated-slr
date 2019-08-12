from torch.utils import data
from constants import InputType, SamplingType, SplitType, SPLIT_ROOT
from abc import abstractmethod

import numpy as np
import os
import cv2 as cv


class AbstractDataset(data.Dataset):

    def __init__(self,
                 root_dir,
                 clip_length,
                 dataset='',
                 split=None,
                 modality=None,
                 sampling=None,
                 transform=None,
                 frame_size=None):
        self.dataset = dataset
        self.split = split
        self.root_dir = root_dir

        self.videos, self.labels = self.load_split()
        self.label2idx = self.load_class_indices()
        self.num_classes = len(self.label2idx.keys())

        self.modality = modality
        self.sampling = sampling
        self.transform = transform
        self.frame_height, self.frame_width = frame_size[0], frame_size[1]
        self.clip_length = clip_length

        self.print_summary()
        
    def print_summary(self):
        print('Initializing dataset: {}, split {}'.format(self.dataset, self.split.value))
        print('Dataset path: {}'.format(self.root_dir))
        print('Number of classes: {}'.format(self.num_classes))
        print()
        print('Input modality: {}'.format(self.modality))
        print('Input sampling: {}'.format(self.sampling))
        print('Input clip length: {}'.format(self.clip_length))
        print('Input transforms: ')
        for t in self.transform.transforms:
            print('\t' + type(t).__name__ + ': ', end='')
            print(vars(t))
        print('\n==========================================')

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int):
        frame_indices = self.get_segment_indices(self.videos[index])

        frames = self.load_frames(self.videos[index], frame_indices)
        label = self.labels[index]

        frames = self.transform(frames)

        return frames, label

    @abstractmethod
    def get_segment_indices(self, video_path):
        pass

    def load_frames(self, video_path, indices):
        frame_tensor = []
        if self.modality == InputType.RGB.value:
            frames = sorted([os.path.join(video_path, '{:06d}.jpg'.format(idx)) for idx in indices])
            frame_tensor = np.empty((len(frames), self.frame_height, self.frame_width, 3), dtype=np.float64)
            for idx, f in enumerate(frames):
                frame_tensor[idx] = np.array(cv.imread(f)).astype(np.float64)

        elif self.modality == InputType.RGB.value:
            raise NotImplementedError

        elif self.modality == InputType.RGB.value:
            raise NotImplementedError

        return frame_tensor

    def load_split(self):
        data = []
        labels = []
        with open(os.path.join(SPLIT_ROOT, self.dataset, self.split.value + '.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                file, lbl = l[:-1].split()
                data.append(os.path.join(self.root_dir, file))
                labels.append(int(lbl))

        return data, labels

    def load_class_indices(self):
        class_indices = {}
        with open(os.path.join(SPLIT_ROOT, self.dataset, 'class_indices.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                idx, lbl, name = l[:-1].split(' ', 2)
                class_indices[idx] = (lbl, name)

        return class_indices
