"""EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.
https://arxiv.org/pdf/2303.14535.pdf
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import albumentations as A
import cv2
import numpy as np
import torch
import tqdm
import os
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.models.components import AnomalyModule

from .torch_model import SegADModel

logger = logging.getLogger(__name__)


class SegAD(AnomalyModule):
    """PL Lightning Module for the EfficientAD algorithm.

    Args:
        teacher_file_name (str): path to the pre-trained teacher model
        teacher_out_channels (int): number of convolution output channels
        image_size (tuple): size of input images
        model_size (str): size of student and teacher model
        lr (float): learning rate
        weight_decay (float): optimizer weight decay
        padding (bool): use padding in convoluional layers
        batch_size (int): batch size for imagenet dataloader
    """

    def __init__(
        self,
        n_clusters: int,
        background_indices: list[int],
        image_size: tuple[int, int],
        seg_dir: str,
        lr: float = 0.0001,
        weight_decay: float = 0.00001,
        padding: bool = False,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.n_clusters = n_clusters
        self.model: SegADModel = SegADModel(
            out_channels=self.n_clusters,
            input_size=self.image_size,
            padding=padding
        )
        self.seg_dir = seg_dir
        self.lr = lr
        self.weight_decay = weight_decay

        self.segs: Dict[str, torch.Tensor] = dict()
        self.background_indices = background_indices
        self.prepare_seg()

    def read_seg_files(self, heat_dir):
        seg = []
        for idx in range(self.n_clusters):
            cluster_file = heat_dir / f'heatresult{idx}.jpg'
            cluster_img = torch.Tensor(cv2.imread(str(cluster_file), 0),
                                       device=self.device)
            seg.append(cluster_img)
        seg = torch.stack(seg, dim=0) / 128.
        # seg = seg.argmax(dim=0)
        # background_mask = []
        # for idx in self.background_indices:
        #     background_mask.append(seg == idx)
        # seg[torch.stack(background_mask, 0).any(0)] = 0
        return seg

    def prepare_seg(self):
        seg_dir = Path(self.seg_dir)
        train_dir = seg_dir / 'train'
        test_dir = seg_dir / 'test'
        for heat_dir in train_dir.iterdir():
            img_name = 'train/good/{:03d}.png'.format(int(heat_dir.name))
            self.segs[img_name] = self.read_seg_files(heat_dir)

        for img_type in ['good', 'logical_anomalies', 'structural_anomalies']:
            test_subdir = test_dir / img_type
            for heat_dir in test_subdir.iterdir():
                img_name = 'test/{}/{:03d}.png'.format(
                    img_type, int(heat_dir.name))
                self.segs[img_name] = self.read_seg_files(heat_dir)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        num_steps = max(
            self.trainer.max_steps, self.trainer.max_epochs *
            len(self.trainer.datamodule.train_dataloader())
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=int(0.95 * num_steps), gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_seg_batch(self, batch: dict[str, str]) -> torch.Tensor:
        seg_batch = []
        for img_path in batch['image_path']:
            img_name = '/'.join(img_path.split('/')[-3:])
            seg_batch.append(self.segs[img_name])
        seg_batch = torch.stack(seg_batch, dim=0)
        return seg_batch

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> dict[str, Tensor]:
        """Training step for EfficintAD returns the  student, autoencoder and combined loss.

        Args:
            batch (batch: dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
          Loss.
        """
        del args, kwargs  # These variables are not used.

        seg_batch = self.get_seg_batch(batch).to(self.device)
        loss_ae, loss_stae = self.model(batch=batch["image"], seg_batch=seg_batch)
        loss = loss_ae + loss_stae

        self.log("ae_loss", loss_ae.item(), on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("stae_loss", loss_stae.item(), on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("train_loss", loss.item(), on_epoch=True,
                 prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation Step of EfficientAD returns anomaly maps for the input image batch

        Args:
          batch (dict[str, str | Tensor]): Input batch

        Returns:
          Dictionary containing anomaly maps.
        """
        del args, kwargs  # These variables are not used.

        seg_batch = self.get_seg_batch(batch).to(self.device)
        batch["anomaly_maps"] = self.model(batch["image"], seg_batch)

        return batch


class SegadLightning(SegAD):
    """PL Lightning Module for the EfficientAD Algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            n_clusters=hparams.model.n_clusters,
            seg_dir=os.path.join(
                hparams.model.seg_dir,
                hparams.dataset.category
            ),
            background_indices=hparams.model.background_indices,
            lr=hparams.model.lr,
            weight_decay=hparams.model.weight_decay,
            padding=hparams.model.padding,
            image_size=hparams.dataset.image_size,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
