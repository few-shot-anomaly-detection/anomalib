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

from typing import Union

import torch
import torchvision
from omegaconf import DictConfig, ListConfig
from torch import nn

from anomalib.models.components.feature_extractors.feature_extractor import FeatureExtractor
from anomalib.models.fastflow.utils import fastflow_head
from anomalib.models.fastflow.anomaly_map import AnomalyMapGenerator


class FastflowModel(nn.Module):
    """FASTFLOW"""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()
        dims = [64, 32, 16]

        self.backbone = getattr(torchvision.models, hparams.model.backbone)
        self.condition_vector: int = hparams.model.condition_vector
        self.dec_arch = hparams.model.decoder
        self.pool_layers = hparams.model.layers

        self.encoder = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.pool_layers)
        self.pool_dims = self.encoder.out_dims
        self.decoders = nn.ModuleList(
            [
                fastflow_head(self.condition_vector, hparams.model.coupling_blocks,
                              hparams.model.clamp_alpha, pool_dim, dim)
                for pool_dim, dim in zip(self.pool_dims, dims)
            ]
        )

        # encoder model is fixed
        for parameters in self.encoder.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator(
            image_size=tuple(hparams.model.input_size), pool_layers=self.pool_layers
        )

    def forward(self, images):
        """Forward-pass images into the network to extract encoder features and compute probability.

        Args:
          images: Batch of images.

        Returns:
          Predicted anomaly maps.

        """

        activation = self.encoder(images)

        distribution = []

        height: list[int] = []
        width: list[int] = []
        avg_loss = 0
        for layer_idx, layer in enumerate(self.pool_layers):
            encoder_activations = activation[layer].detach()  # bxcxhxw

            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            height.append(im_height)
            width.append(im_width)
            decoder = self.decoders[layer_idx].to(images.device)
            # decoder returns the transformed variable z and the log jacobian determinant
            p_u, log_jac_det = decoder(encoder_activations)

            log_prob = -torch.mean(p_u ** 2, dim=1, keepdim=True) * 0.5
            distribution.append(torch.exp(log_prob).detach())
            if self.training:
                decoder_log_prob = torch.mean(0.5 * torch.sum(p_u ** 2, dim=(1, 2, 3)) - log_jac_det)
                avg_loss += decoder_log_prob

        if self.training:
            return avg_loss
        else:
            return self.anomaly_map_generator(distribution=distribution, height=height, width=width).to(images.device)
