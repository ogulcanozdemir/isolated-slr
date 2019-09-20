import torch


class ClipToTensor(object):

    def __init__(self, div=True):
        self.div = div

    def __call__(self, sequence):
        sequence = torch.from_numpy(sequence).permute(0, 1)
        return sequence.float() # .div(255) if self.div else frame_group.float()