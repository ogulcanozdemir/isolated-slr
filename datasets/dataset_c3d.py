from datasets.dataset import GenericDataset
from utils.constants import SamplingType
from utils.logger import logger
from utils.video_transforms import ClipSubtractMean

import numpy as np
import os


class DatasetC3D(GenericDataset):

    # paper parameters
    # resize_height = 128
    # resize_width = 171
    # crop_size = 112

    crop_size = 112

    def __init__(self,
                 root_dir,
                 dataset='',
                 test_mode=False,
                 split=None,
                 modality=None,
                 sampling=None,
                 frame_size=None,
                 transform=None,
                 clip_length=16,
                 feature_extract=False):
        super(DatasetC3D, self).__init__(root_dir=root_dir,
                                         dataset=dataset,
                                         split=split,
                                         modality=modality,
                                         sampling=sampling,
                                         transform=transform,
                                         frame_size=frame_size,
                                         clip_length=clip_length,
                                         test_mode=test_mode,
                                         feature_extract=feature_extract)

        self.print_summary()

    def print_summary(self):
        logger.info('Initializing dataset: {}, split {}'.format(self.dataset, self.split.value))
        logger.info('Dataset path: {}'.format(self.root_dir))
        logger.info('Number of classes: {}'.format(self.num_classes))
        logger.info('')
        logger.info('Input frame size: {}x{}'.format(self.frame_height, self.frame_width))
        logger.info('Input modality: {}'.format(self.modality))
        logger.info('Input sampling: {}'.format(self.sampling))
        logger.info('Input clip length: {}'.format(self.clip_length))
        logger.info('Input transforms: ')
        for t in self.transform.transforms:
            if isinstance(t, ClipSubtractMean):
                logger.info('\t' + type(t).__name__ + ': True')
            else:
                logger.info('\t' + type(t).__name__ + ': ' + str(vars(t)))
        logger.info('==========================================')

    def __getitem__(self, index: int):
        if not self.feature_extract:
            frame_indices = self.get_segment_indices(self.videos[index])

            frames = self.load_frames(self.videos[index], frame_indices)
            label = self.labels[index]

            frames = self.transform(frames)

            if self.test_mode:
                return frames, label, self.videos[index]
            else:
                return frames, label
        else:
            frame_indices_list = self.get_segment_indices(self.videos[index])

            frame_list = []
            for frame_indices in frame_indices_list:
                frames = self.load_frames(self.videos[index], frame_indices)
                frames = self.transform(frames)
                frame_list.append(frames)

            return frame_list, self.videos[index]

    def get_segment_indices(self, video_path):
        frame_list = sorted(os.listdir(video_path))
        frame_count = len(frame_list)

        segment_indices = []
        if self.sampling == SamplingType.RANDOM.value:
            # randomly select start frame for a volume with self.clip_length
            time_idx = np.random.randint(frame_count - self.clip_length)
            segment_indices = list(np.arange(time_idx, time_idx + self.clip_length))

        elif self.sampling == SamplingType.ALL.value:
            # return all of frames
            segment_indices = list(range(0, frame_count))

        elif self.sampling == SamplingType.KEYFRAME.value:
            # return keyframes
            raise NotImplementedError

        elif self.sampling == SamplingType.EQUIDISTANT.value:
            # return equally spaced frames
            segment_indices = np.linspace(0, frame_count-1, self.clip_length, dtype=np.int)

        elif self.sampling == SamplingType.FEATURE16_NONOVERLAP.value:
            segment_indices = []
            idx = 0
            while idx <= frame_count - 1:
                diff = 16
                if idx + 16 >= frame_count - 1:
                    diff = frame_count - idx

                segment_indices.append(np.arange(idx, idx+diff))
                idx += 16

            # add overlapped frame if the size of last segment is less than 16 frames
            if len(segment_indices[-1]) < 16:
                extract_len = 16 - (len(segment_indices[-2]) - len(segment_indices[-1]))
                last_segment_start_idx = segment_indices[-2][extract_len]
                segment_indices[-1] = np.arange(last_segment_start_idx, frame_count)

            # frame_indices = np.arange(0, frame_count)
            # num_sections = np.ceil(frame_indices.shape[0] // 16).astype(np.int)
            # segment_indices = np.array_split(frame_indices, num_sections)

        elif self.sampling == SamplingType.FEATURE16_OVERLAP.value:
            frame_indices = np.linspace(0, frame_count-16, self.clip_length, dtype=np.int)
            segment_indices = []
            for idx in frame_indices:
                segment_indices.append(np.arange(idx, idx+16))

        return segment_indices

