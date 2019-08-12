from datasets.dataset import AbstractDataset
from constants import InputType,SamplingType, SplitType

import numpy as np
import os


class DatasetC3D(AbstractDataset):

    # paper parameters
    # resize_height = 128
    # resize_width = 171
    # crop_size = 112

    frame_height = 150
    frame_width = 150
    crop_size = 112

    def __init__(self,
                 root_dir,
                 dataset='',
                 split=None,
                 modality=None,
                 sampling=None,
                 transform=None,
                 clip_length=16):
        super(DatasetC3D, self).__init__(root_dir=root_dir,
                                         dataset=dataset,
                                         split=split,
                                         modality=modality,
                                         sampling=sampling,
                                         transform=transform,
                                         frame_size=(self.frame_height, self.frame_width),
                                         clip_length=clip_length)

    def get_segment_indices(self, video_path):
        frame_list = sorted(os.listdir(video_path))
        frame_count = len(frame_list)

        segment_indices = []
        if self.sampling == SamplingType.RANDOM.value:
            # randomly select start frame for a volume with self.clip_length
            time_idx = np.random.randint(frame_count - self.clip_length)
            segment_indices = list(np.arange(time_idx, time_idx + self.clip_length))

        elif self.sampling == SamplingType.RANDOM.value:
            # return all of frames
            segment_indices = list(range(0, frame_count))

        elif self.sampling == SamplingType.RANDOM.value:
            # return keyframes
            raise NotImplementedError

        elif self.sampling == SamplingType.RANDOM.value:
            # return equally spaced frames
            segment_indices = np.linspace(0, frame_count-1, self.clip_length, dtype=np.int)

        return segment_indices

