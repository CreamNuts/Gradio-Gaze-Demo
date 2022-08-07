from typing import Tuple, Union

import numpy as np
import torch


class Bucketizer:
    def __init__(self, range: Tuple[float, float], num_bins: int):
        self.range = range
        self.num_bins = num_bins
        self.bins = torch.linspace(range[0], range[1], num_bins)

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        assert (data >= self.range[0]).all() and (data <= self.range[1]).all(), "data out of range"
        return torch.bucketize(data, self.bins, right=True)

    def unbucketize(self, data):
        return self.bins[data]
