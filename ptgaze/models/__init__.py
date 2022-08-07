from omegaconf import DictConfig
from torch import nn

from .bucketizer import Bucketizer


def create_bucketizer(config: DictConfig):
    return Bucketizer(range=config.bucketizer.range, num_bins=config.bucketizer.num_bins)


def create_model(
    config: DictConfig,
):
    """
    Create a model from a model name.
    """
    from timm import create_model

    net = create_model(model_name=config.model.name)
    if hasattr(net, "fc"):
        net.fc = nn.Linear(
            net.fc.in_features, config.bucketizer.num_bins * config.bucketizer.num_chunks
        )
    elif hasattr(net, "head"):
        net.head = nn.Linear(
            net.head.in_features, config.bucketizer.num_bins * config.bucketizer.num_chunks
        )
    return net
