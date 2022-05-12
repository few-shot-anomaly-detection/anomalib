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

from typing import List, Tuple, Union, cast, Dict

import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor, nn, optim
from einops import rearrange

from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.models.components.feature_extractors.feature_extractor import FeatureExtractor
from anomalib.models.diffusion.backbone import Unet, GaussianDiffusion, MLP

__all__ = ["AnomalyMapGenerator", "DiffusionModel", "DiffusionLightning"]


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        image_size: Union[ListConfig, Tuple],
        pool_layers: List[str],
    ):
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.pool_layers: List[str] = pool_layers

    def compute_anomaly_map(
        self, nll: Union[List[Tensor], List[List]], height: List[int], width: List[int]
    ) -> Tensor:
        """Compute the layer map based on likelihood estimation.

        Args:
          distribution: Probability distribution for each decoder block
          height: blocks height
          width: blocks width

        Returns:
          Final Anomaly Map

        """
        test_map: List[Tensor] = []
        n_layers = len(self.pool_layers)
        for layer_idx in range(n_layers):
            test_logp = -nll[layer_idx].clone().double()
            # test_norm -= torch.max(test_norm)  # normalize likelihoods to (-Inf:0] by subtracting a constant
            test_prob = torch.exp(test_logp)  # convert to probs in range [0:1]
            test_mask = test_prob.reshape(-1, height[layer_idx], width[layer_idx])
            # upsample
            test_mask = F.interpolate(
                test_mask.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=True
            ).squeeze()
            test_mask = torch.log(test_mask)
            test_map.append(
                test_mask
            )
        anomaly_map_list = torch.stack(test_map, dim=-1)
        # invert probs to anomaly scores
        anomaly_map = -torch.mean(anomaly_map_list, dim=-1)
        # anomaly_map = torch.abs(anomaly_map - torch.mean(anomaly_map, dim=0,
        #                         keepdim=True) / torch.std(anomaly_map, dim=0))

        return anomaly_map

    def __call__(self, **kwargs: Union[List[Tensor], List[int], List[List]]) -> Tensor:
        """Returns anomaly_map.

        Expects `distribution`, `height` and 'width' keywords to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.input_size),
        >>>        pool_layers=pool_layers)
        >>> output = self.anomaly_map_generator(distribution=dist, height=height, width=width)

        Raises:
            ValueError: `distribution`, `height` and 'width' keys are not found

        Returns:
            torch.Tensor: anomaly map
        """
        if not ("distribution" in kwargs and "height" in kwargs and "width" in kwargs):
            raise KeyError(f"Expected keys `distribution`, `height` and `width`. Found {kwargs.keys()}")

        # placate mypy
        distribution: List[Tensor] = cast(List[Tensor], kwargs["distribution"])
        height: List[int] = cast(List[int], kwargs["height"])
        width: List[int] = cast(List[int], kwargs["width"])
        return self.compute_anomaly_map(distribution, height, width)


class DiffusionModel(nn.Module):
    """FASTFLOW"""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()

        self.backbone = getattr(torchvision.models, hparams.model.backbone)
        self.pool_layers = hparams.model.layers

        self.encoder = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.pool_layers)
        self.pool_dims = self.encoder.out_dims
        print(self.pool_dims)
        self.dim_mults = hparams.model.dim_mults

        pool_size = 64
        self.pool = lambda x: F.avg_pool1d(
            x, pool_size
        )
        self.decoders = nn.ModuleList(
            [
                GaussianDiffusion(
                    # MLP(pool_dim, pool_dim),
                    # Unet(dim=pool_dim, dim_mults=self.dim_mults, channels=pool_dim),
                    Unet(dim=int(pool_dim / pool_size) * 8,
                         dim_mults=self.dim_mults,
                         channels=int(pool_dim / pool_size)),
                    timesteps=1000, loss_type='l1'
                )
                for pool_dim in self.pool_dims
            ]
        )

        # encoder model is fixed
        for parameters in self.encoder.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator(
            image_size=tuple(hparams.model.input_size), pool_layers=self.pool_layers
        )

    def forward(self, images: Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward-pass images into the network to extract encoder features and compute probability.

        Args:
          images: Batch of images.

        Returns:
          Predicted anomaly maps.

        """
        activation = self.encoder(images)

        nlls = []

        height: list[int] = []
        width: list[int] = []
        for layer_idx, layer in enumerate(self.pool_layers):
            encoder_activations = activation[layer].detach()  # bxcxhxw
            n, c, h, w = encoder_activations.size()
            encoder_activations = rearrange(
                self.pool(rearrange(encoder_activations, 'n c w h -> n (w h) c')),
                'n (w h) c -> n c w h',
                n=n, h=h, w=h
            )
            # encoder_activations = torch.mean(encoder_activations, dim=1, keepdim=True)

            batch_size, _, im_height, im_width = encoder_activations.size()

            height.append(im_height)
            width.append(im_width)
            decoder = self.decoders[layer_idx]
            # decoder returns the transformed variable z and the log jacobian determinant
            n_mcmc = 10
            for i in range(n_mcmc):
                tmp = decoder.log_prob(encoder_activations, inter_steps=10) / n_mcmc
                if i == 0:
                    nll = tmp
                else:
                    nll += tmp
            nll = torch.mean(nll, dim=1)

            nlls.append(nll.detach())

        output = self.anomaly_map_generator(distribution=nlls, height=height, width=width)
        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> torch.Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: Dict[str:Tensor]:

        Returns:
            Embedding vector
        """

        embeddings = torch.mean(features[self.pool_layers[0]], dim=1, keepdim=True)
        for layer in self.pool_layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            layer_embedding = torch.mean(layer_embedding, dim=1, keepdim=True)
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings


class DiffusionLightning(AnomalyModule):
    """PL Lightning Module for the Diffusion algorithm."""

    def __init__(self, hparams):
        super().__init__(hparams)
        self.model: DiffusionModel = DiffusionModel(hparams)

    def configure_callbacks(self):
        """Configure model-specific callbacks."""
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(list(self.model.decoders[decoder_idx].parameters()))

        optimizer = optim.Adam(
            params=decoders_parameters,
            lr=self.hparams.model.lr,
        )
        return optimizer

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of diffusion model.

        For each batch, decoder layers are trained with a dynamic fiber batch size.
        Training step is performed manually as multiple training steps are involved
            per batch of input images

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Loss value for the batch

        """
        images = batch["image"]
        activation = self.model.encoder(images)
        avg_loss = torch.zeros([1], dtype=torch.float64, device=self.device)

        for layer_idx, layer in enumerate(self.model.pool_layers):
            encoder_activations = activation[layer].detach()  # BxCxHxW
            n, c, h, w = encoder_activations.size()
            encoder_activations = rearrange(
                self.model.pool(rearrange(encoder_activations, 'n c w h -> n (w h) c')),
                'n (w h) c -> n c w h',
                n=n, h=h, w=h
            )
            # encoder_activations = torch.mean(encoder_activations, dim=1, keepdim=True)

            decoder = self.model.decoders[layer_idx]
            loss = decoder(encoder_activations)
            avg_loss += loss

        avg_loss /= len(self.model.pool_layers)
        self.log('ddpm loss', avg_loss)
        return {"loss": avg_loss}

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of CFLOW.

            Similar to the training step, encoder features
            are extracted from the CNN for each batch, and anomaly
            map is computed.

        Args:
          batch: Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.

        """
        batch["anomaly_maps"] = self.model(batch["image"])

        return batch
