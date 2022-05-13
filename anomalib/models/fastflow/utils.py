"""FASTFLOW: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.

[FASTFLOW-AD](https://arxiv.org/pdf/2111.07677.pdf)
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from torch import nn
import logging

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework.sequence_inn import SequenceINN

logger = logging.getLogger(__name__)


def subnet_conv1(dims_in: int, dims_out: int):
    """1x1 conv subnetwork to predicts the affine coefficients.

    Args:
        dims_in (int): input dimensions
        dims_out (int): output dimensions

    Returns:
        nn.Sequential: Feed-forward subnetwork
    """
    kernel_size = 1
    return nn.Sequential(
        nn.Conv2d(dims_in, 2 * dims_in, kernel_size=kernel_size),
        nn.ReLU(),
        nn.Conv2d(2 * dims_in, dims_out, kernel_size=kernel_size)
    )


def subnet_conv3(dims_in: int, dims_out: int):
    """3x3 conv subnetwork to predicts the affine coefficients.

    Args:
        dims_in (int): input dimensions
        dims_out (int): output dimensions

    Returns:
        nn.Sequential: Feed-forward subnetwork
    """
    kernel_size = 3
    return nn.Sequential(
        nn.Conv2d(dims_in, 2 * dims_in, kernel_size=kernel_size, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * dims_in, dims_out, kernel_size=kernel_size, padding=1)
    )


def subnet_conv_func(kernel_size, hidden_ratio=1.0):
    def subnet_conv(dims_in: int, dims_out: int):
        dims_hidden = int(dims_in * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(dims_in, dims_hidden, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.Conv2d(dims_hidden, dims_out, kernel_size=kernel_size, padding='same')
        )
    return subnet_conv


def fastflow_head(condition_vector: int, coupling_blocks: int, clamp_alpha: float, n_features: int, dim) -> SequenceINN:
    """Create invertible decoder network.

    Args:
        condition_vector (int): length of the condition vector
        coupling_blocks (int): number of coupling blocks to build the decoder
        clamp_alpha (float): clamping value to avoid exploding values
        n_features (int): number of decoder features

    Returns:
        SequenceINN: decoder network block
    """
    coder = Ff.SequenceINN(n_features, dim, dim)
    logger.info("CNF coder: %d, %d", n_features, dim)
    for i in range(coupling_blocks):
        if i % 2 == 1:
            kernel_size = 1
        else:
            kernel_size = 3
        coder.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size),
            affine_clamping=clamp_alpha,
            permute_soft=False
        )
        # coder.append(
        #     Fm.AllInOneBlock,
        #     subnet_constructor=subnet_conv3,
        #     affine_clamping=clamp_alpha,
        #     global_affine_type="SOFTPLUS",
        # )
        # coder.append(
        #     Fm.AllInOneBlock,
        #     subnet_constructor=subnet_conv1,
        #     affine_clamping=clamp_alpha,
        #     global_affine_type="SOFTPLUS",
        # )
    return coder
