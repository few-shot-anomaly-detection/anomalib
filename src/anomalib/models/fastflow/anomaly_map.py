"""FastFlow Anomaly Map Generator Implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: list[Tensor]) -> Tensor:
        """Generate Anomaly Heatmap.

        This implementation generates the heatmap based on the flow maps
        computed from the normalizing flow (NF) FastFlow blocks. Each block
        yields a flow map, which overall is stacked and averaged to an anomaly
        map.

        Args:
            hidden_variables (list[Tensor]): List of hidden variables from each NF FastFlow block.

        Returns:
            Tensor: Anomaly Map.
        """
        flow_maps: list[Tensor] = []
        img_probs: list[Tensor] = []
        for hidden_variable in hidden_variables:
            # NOTE
            # n, _, w, h = hidden_variable.shape
            # hidden_variables = hidden_variable.view(n, w, h, -1).permute(0, 3, 1, 2)

            log_prob = -torch.mean(hidden_variable**2, dim=1, keepdim=True) * 0.5
            img_probs.append(-log_prob.reshape(log_prob.shape[0], -1).sum(dim=1))
            prob = torch.exp(log_prob)
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        flow_maps = torch.stack(flow_maps, dim=-1)
        anomaly_map = torch.mean(flow_maps, dim=-1)

        # log_probs = torch.stack(img_probs, dim=-1)
        # anomaly_score = torch.max(log_probs, dim=-1).values
        anomaly_score = anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values

        return anomaly_map, anomaly_score
