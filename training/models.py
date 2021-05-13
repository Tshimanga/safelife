import numpy as np

from torch import nn
from torch.nn import functional as F

from .global_config import HyperParam, update_hyperparams


class SE_Block(nn.Module):
    """credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py"""
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


def conv_se_block(in_c, out_c, se_reduction=None, *args, **kwargs):
    if se_reduction is not None:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, *args, **kwargs),
            nn.BatchNorm2d(out_c),
            SE_Block(out_c, r=se_reduction),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, *args, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )


def safelife_cnn_se(input_shape, se_reduction=8):
    """
    Defines a CNN with good default values for safelife.

    This works best for inputs of size 25x25.

    Parameters
    ----------
    input_shape : tuple of ints
        Height, width, and number of channels for the board.

    Returns
    -------
    cnn : torch.nn.Sequential
    output_shape : tuple of ints
        Channels, width, and height.

    Returns both the CNN module and the final output shape.
    """
    h, w, c = input_shape
    cnn = nn.Sequential(
        conv_se_block(c, 32, se_reduction=se_reduction, kernel_size=5, stride=2),
        conv_se_block(32, 64, se_reduction=se_reduction, kernel_size=3, stride=2),
        conv_se_block(64, 64, se_reduction=None, kernel_size=3, stride=1),
    )
    h = (h-4+1)//2
    h = (h-2+1)//2
    h = (h-2)
    w = (w-4+1)//2
    w = (w-2+1)//2
    w = (w-2)
    return cnn, (64, w, h)


def safelife_cnn(input_shape):
    """
    Defines a CNN with good default values for safelife.

    This works best for inputs of size 25x25.

    Parameters
    ----------
    input_shape : tuple of ints
        Height, width, and number of channels for the board.

    Returns
    -------
    cnn : torch.nn.Sequential
    output_shape : tuple of ints
        Channels, width, and height.

    Returns both the CNN module and the final output shape.
    """
    h, w, c = input_shape
    cnn = nn.Sequential(
        nn.Conv2d(c, 32, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU()
    )
    h = (h-4+1)//2
    h = (h-2+1)//2
    h = (h-2)
    w = (w-4+1)//2
    w = (w-2+1)//2
    w = (w-2)
    return cnn, (64, w, h)


class SafeLifeQNetwork(nn.Module):
    """
    Module for calculating Q functions.
    """
    def __init__(self, input_shape):
        super().__init__()

        self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        self.advantages = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.value_func = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs).flatten(start_dim=1)
        advantages = self.advantages(x)
        value = self.value_func(x)
        qval = value + advantages - advantages.mean()
        return qval


@update_hyperparams
class SafeLifePolicyNetwork(nn.Module):

    dense_depth: HyperParam = 1
    dense_width: HyperParam = 512

    def __init__(self, input_shape, r=16):
        super().__init__()

        # self.cnn, cnn_out_shape = safelife_cnn(input_shape)
        self.cnn, cnn_out_shape = safelife_cnn_se(input_shape, se_reduction=r)
        num_features = np.product(cnn_out_shape)
        num_actions = 9

        self.dropout = nn.Dropout(0.25)

        dense = [nn.Sequential(nn.Linear(num_features, self.dense_width), nn.ReLU())]
        for n in range(self.dense_depth - 1):
            dense.append(nn.Sequential(nn.Linear(self.dense_width, self.dense_width), nn.ReLU()))
        self.dense = nn.Sequential(*dense)

        self.logits = nn.Linear(self.dense_width, num_actions)
        self.value_func = nn.Linear(self.dense_width, 1)

    def forward(self, obs):
        # Switch observation to (c, w, h) instead of (h, w, c)
        obs = obs.transpose(-1, -3)
        x = self.cnn(obs)
        # x = self.se(x)  # try applying it before relu after bn
        x = x.flatten(start_dim=1)
        for layer in self.dense:
            x = layer(x)
        # x = self.dropout(x)
        value = self.value_func(x)[...,0]
        policy = F.softmax(self.logits(x), dim=-1)
        return value, policy
