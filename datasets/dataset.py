from torch.utils import data
from abc import abstractmethod
from utils.constants import *

import numpy as np
import os
import cv2 as cv


class GenericDataset(data.Dataset):

    def __init__(self,
                 root_dir,
                 clip_length=None,
                 dataset='',
                 split=None,
                 modality=None,
                 sampling=None,
                 transform=None,
                 frame_size=None,
                 test_mode=False,
                 feature_extract=False):
        self.dataset = dataset
        self.split = split
        self.root_dir = root_dir

        self.videos, self.labels = self.load_split()
        self.label2idx = self.load_class_indices()
        self.num_classes = len(self.label2idx.keys())

        self.modality = modality
        self.sampling = sampling
        self.transform = transform
        if frame_size and clip_length:
            self.frame_height, self.frame_width = int(frame_size.split('x')[0]), int(frame_size.split('x')[1]),   # h, w
            self.clip_length = clip_length
        self.test_mode = test_mode
        self.feature_extract = feature_extract

    @abstractmethod
    def print_summary(self):
        pass

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int):
        pass

    @abstractmethod
    def get_segment_indices(self, video_path):
        pass

    def load_frames(self, video_path, indices):
        frame_tensor = []
        if self.modality == InputType.RGB.value:
            frames = sorted([os.path.join(video_path, '{:06d}.jpg'.format(idx)) for idx in indices])
            frame_tensor = np.empty((len(frames), self.frame_height, self.frame_width, 3), dtype=np.float64)
            for idx, f in enumerate(frames):
                img = cv.imread(f)
                if img.shape[0] != self.frame_height and img.shape[1] != self.frame_width:
                    img = cv.resize(img, (self.frame_height, self.frame_width))
                # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                frame_tensor[idx] = np.array(img).astype(np.float64)

        elif self.modality == InputType.FLOW.value:
            raise NotImplementedError

        elif self.modality == InputType.DEPTH.value:
            raise NotImplementedError

        return frame_tensor

    def load_sequence(self, sequence_path):
        sequence_tensor = np.load(os.path.join(sequence_path + '.npz'))['arr_0']
        return sequence_tensor

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
