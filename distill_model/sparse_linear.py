# sparse_linear.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLinear(nn.Module):
    """
    Linear layer with fixed weight sparsity and straight-through estimation.

    Given a target sparsity ratio, the layer computes a binary mask that zeros
    out the weights with the smallest absolute magnitude. The mask is fixed
    (not learned) and stored as a buffer. During the forward pass the mask is
    applied to the weights, but gradients flow through to *all* weight entries
    via straight-through estimation so that the dense weight tensor can be
    updated by the optimizer.

    Args:
        in_features:  Size of each input sample.
        out_features: Size of each output sample.
        bias:         If ``True``, the layer has a learnable bias.
        target_sparsity: Fraction of weights to zero out, in [0, 1).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        target_sparsity: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.target_sparsity = target_sparsity

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Binary mask: True = keep, False = prune.  Stored as a non-trainable
        # buffer so it is automatically moved across devices / saved in state
        # dicts but never updated by the optimizer.
        self.register_buffer(
            'weight_mask',
            torch.ones(out_features, in_features, dtype=torch.bool),
        )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def compute_mask(self) -> None:
        """Recompute ``weight_mask`` from the current weights.

        Weights with the smallest absolute value are masked out until the
        desired ``target_sparsity`` fraction is reached.
        """
        if self.target_sparsity <= 0.0:
            self.weight_mask.fill_(True)
            return

        w_abs = self.weight.abs().view(-1)
        n_total = w_abs.numel()
        n_prune = int(n_total * self.target_sparsity)

        if n_prune == 0:
            self.weight_mask.fill_(True)
            return

        threshold = torch.kthvalue(w_abs, n_prune).values
        self.weight_mask.copy_(w_abs.view_as(self.weight_mask) > threshold)

    @classmethod
    def from_pretrained(
        cls,
        linear: nn.Linear,
        target_sparsity: float = 0.0,
    ) -> SparseLinear:
        """Create a ``SparseLinear`` from an existing ``nn.Linear``.

        The pretrained weights (and bias, if present) are copied and a
        magnitude-based sparsity mask is computed immediately.

        Args:
            linear:          A pretrained ``nn.Linear`` layer.
            target_sparsity: Fraction of weights to prune, in [0, 1).

        Returns:
            A new ``SparseLinear`` instance ready for fine-tuning.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            target_sparsity=target_sparsity,
        )

        layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)

        layer.compute_mask()
        return layer

    # ------------------------------------------------------------------
    # Forward with straight-through estimation
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Straight-through estimator:
        #   forward value : weight * mask   (sparse)
        #   backward grad : d L / d weight  (dense, as if the mask were absent)
        #
        # We achieve this with the identity:
        #   masked_weight = weight - (weight * ~mask).detach()
        # The detached term has the correct value but contributes no gradient,
        # so the autograd graph sees the gradient flowing through `weight`
        # unchanged while the forward output is correctly masked.
        masked_weight = self.weight - (self.weight * ~self.weight_mask).detach()
        return F.linear(x, masked_weight, self.bias)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}, '
            f'target_sparsity={self.target_sparsity}'
        )
