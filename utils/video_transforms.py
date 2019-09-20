import numpy as np
import random
import numbers
import torch
import cv2


class Compose(object):
    """ Composes several transforms together. """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_group):
        for t in self.transforms:
            frame_group = t(frame_group)
        return frame_group


class ClipResize(object):

    def __init__(self, size=112):
        if isinstance(size, numbers.Number):
            self.size = (size, size)

    def __call__(self, frame_group):
        resized_frames = [cv2.resize(f, self.size, interpolation=cv2.INTER_CUBIC) for f in frame_group]
        return resized_frames


class ClipRandomCrop(object):

    """ Spatially crop given clip (frame group), new size should be smaller than original size"""

    def __init__(self, size=112):
        if isinstance(size, numbers.Number):
            self.size = (size, size)

    def __call__(self, frame_group):
        new_h, new_w = self.size
        h, w, c = frame_group[0].shape

        i = random.randint(0, w - new_w)
        j = random.randint(0, h - new_h)

        # cropped_frame_group = [img[j:j+new_h, i:i+new_w, :] for img in frame_group]
        cropped_frame_group = frame_group[:, j:j+new_h, i:i+new_w, :]

        return cropped_frame_group


class ClipHorizontalFlip(object):

    """ Horizontally flip clips with probability (default 0.5) """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, frame_group):
        if random.random() < self.p:
            frame_group = np.array([np.fliplr(img) for img in frame_group])

        return frame_group


class ClipStandardize(object):

    """ usage for [-1, 1]: video_transforms.ClipStandardize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frame_tensor):
        rep_mean = self.mean * (frame_tensor.size()[0] // len(self.mean))
        rep_std = self.std * (frame_tensor.size()[0] // len(self.std))

        for t, m, s in zip(frame_tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return frame_tensor


class ClipNormalize(object):

    def __init__(self, clip_range=(0, 1)):
        self.clip_range = clip_range

    def __call__(self, frame_group):
        normalized_frame_group = frame_group - frame_group.min()
        normalized_frame_group /= frame_group.max() - frame_group.min()
        return (self.clip_range[1] - self.clip_range[0]) * normalized_frame_group + self.clip_range[0]


class ClipSubtractMean(object):

    def __init__(self, crop_mean):
        self.crop_mean = np.load(crop_mean)

    def __call__(self, frame_group):
        frame_group -= self.crop_mean
        return frame_group


class ClipToTensor(object):

    def __init__(self, div=True):
        self.div = div

    def __call__(self, frame_group):
        frame_group = torch.from_numpy(frame_group).permute(3, 0, 1, 2)
        return frame_group.float() # .div(255) if self.div else frame_group.float()
