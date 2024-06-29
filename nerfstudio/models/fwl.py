"""
Implementation of model with focus-weighted loss function.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig


@dataclass
class FWLModelConfig(NeuSFactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: FWLModel)
    fwl_mult: float = 0.1
    rgb_mult: float=1.0
    """Focus-weighted loss multiplier"""


class FWLModel(NeuSFactoModel):
    """MonoSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: FWLModelConfig

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        image = batch["image"].to(self.device)
        focus_mask = batch["focus_mask"].to(self.device)
        unreduced_loss = self.rgb_loss2(image, outputs["rgb"])
        weighted_loss = unreduced_loss * focus_mask
        fwloss = weighted_loss.mean()
        loss_dict['focus_weighted_loss'] = fwloss * self.config.fwl_mult

        # fwl_mult is actually used on the original rgb loss 

        # dont use the original rgb loss
        loss_dict['rgb_loss'] = loss_dict['rgb_loss'] * self.config.rgb_mult

        return loss_dict
