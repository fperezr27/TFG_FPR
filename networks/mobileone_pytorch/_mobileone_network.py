from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from ._mobileone_block import MobileOneBlock
from ._reparametrizable_module import ReparametrizableModule, ReparametrizableSequential


class MobileOneSize(Enum):
    S0 = "s0"
    S1 = "s1"
    S2 = "s2"
    S3 = "s3"
    S4 = "s4"


@dataclass
class MobileOneConfiguration:
    num_blocks: List[int]
    out_channels: List[int]
    strides: List[int]
    ks: List[int]
    num_classes: int


@dataclass
class NetworkConfig:
    ks: List[int]
    alphas: List[float]


@dataclass
class NetworkBasicConfig:
    num_blocks: List[int]
    strides: List[int]
    out_channels: List[int]


BaseConfig = Dict[str, List[int]]
NetworkConfigs = Dict[MobileOneSize, NetworkConfig]

_BASE_CONFIG = NetworkBasicConfig(
    num_blocks=[1, 2, 8, 5, 5, 1],
    strides=[2, 2, 2, 2, 1, 2],
    out_channels=[64, 64, 128, 256, 256, 512],
)

_NETWORK_CONFIGS: NetworkConfigs = {
    MobileOneSize.S0: NetworkConfig(
        alphas=[0.75, 0.75, 1.0, 1.0, 1.0, 2.0],
        ks=[4, 4, 4, 4, 4, 4],
    ),
    MobileOneSize.S1: NetworkConfig(
        alphas=[1.5, 1.5, 1.5, 2.0, 2.0, 2.5],
        ks=[1, 1, 1, 1, 1, 1],
    ),
    MobileOneSize.S2: NetworkConfig(
        alphas=[1.5, 1.5, 2.0, 2.5, 2.5, 4.0],
        ks=[1, 1, 1, 1, 1, 1],
    ),
    MobileOneSize.S3: NetworkConfig(
        alphas=[2.0, 2.0, 2.5, 3.0, 3.0, 4.0],
        ks=[1, 1, 1, 1, 1, 1],
    ),
    MobileOneSize.S4: NetworkConfig(
        alphas=[3.0, 3.0, 3.5, 3.5, 3.5, 4.0],
        ks=[1, 1, 1, 1, 1, 1],
    ),
}


def get_params(size: MobileOneSize, num_classes: int) -> MobileOneConfiguration:
    conf = _NETWORK_CONFIGS[size]

    out_channels = [
        int(out_ch * alfa)
        for out_ch, alfa in zip(
            _BASE_CONFIG.out_channels,
            conf.alphas,
        )
    ]

    return MobileOneConfiguration(
        num_blocks=_BASE_CONFIG.num_blocks,
        out_channels=out_channels,
        strides=_BASE_CONFIG.strides,
        ks=conf.ks,
        num_classes=num_classes,
    )


class MobileOneNetwork(ReparametrizableModule):
    """MobileOne network.

    Described in detail here: https://arxiv.org/abs/2206.04040
    """

    def __init__(
        self,
        ks: List[int],
        out_channels: List[int],
        num_blocks: List[int],
        strides: List[int],
        num_classes: int = 6,
        gpu_ids=[]
    ):
        super().__init__()
        # self._features = ReparametrizableSequential(
        #     OrderedDict(
        #         [
        #             (
        #                 f"_stage{i+1}",
        #                 _compose_stage(
        #                     num_blocks=num_blocks[i],
        #                     k=ks[i],
        #                     in_channels=51 if i == 0 else out_channels[i - 1],  
        #                     out_channels=out_channels[i],
        #                     stride=strides[i],
        #                 ),
        #             )
        #             for i in range(len(num_blocks))
        #         ]
        #     )
        # )
        # self._average_pooling = nn.AdaptiveAvgPool2d((2, 2))
        # self._linear = nn.Linear(
        #     in_features=out_channels[-1],
        #     out_features=num_classes
        # )
        # self._conv = nn.Conv2d(out_channels[-1],num_classes,kernel_size=3, padding=1)
        # self.cbr_unit = nn.Sequential(self._conv, nn.ReLU(inplace=True))
        self.gpu_ids = gpu_ids
        model = [nn.Conv2d(51, 6, kernel_size=7, padding=3),
                 nn.BatchNorm2d(6, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(6 * mult, 6 * mult * 2, kernel_size = 3,
                                stride=2, padding=1),
                      nn.BatchNorm2d(6 * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(6):
            model += [MobileOneBlock(1,6 * mult,6 * mult,1)]
        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            model += [nn.ConvTranspose2d(6 * mult, int(6 * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.BatchNorm2d(int(6 * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv2d(6,6,kernel_size=7, padding=3)]

        self.model = nn.Sequential(*model)

    @property
    def num_classes(self) -> int:
        return self._linear.out_features

    def forward(self, x: Tensor) -> Tensor:
        # x = self._features(x)
        # x = self._average_pooling(x)
        # #x = torch.flatten(x, 1)
        # #x = self._linear(x)
        # x = self.cbr_unit(x)
        
        #return self.model(x)
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            return self.model(x)

    def reparametrize(self) -> nn.Sequential:
        return nn.Sequential(
            *[
                self._features.reparametrize(),
                self._average_pooling,
                nn.Flatten(),
                self._linear,
            ]
        )


def _compose_stage(
    num_blocks: int,
    k: int,
    in_channels: int,
    out_channels: int,
    stride: int,
) -> ReparametrizableSequential:
    return ReparametrizableSequential(
        *[
            MobileOneBlock(
                k=k,
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
            )
            for i in range(num_blocks)
        ]
    )
