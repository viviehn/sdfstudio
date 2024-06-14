"""
Implementation of model with focus-weighted loss function.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig

def get_focus_weighted_loss(rgb_loss, weight_vals):
    return (rgb_loss*weight_vals.to(rgb_loss.device)).mean()


@dataclass
class FWLModelConfig(NeuSFactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: FWLModel)
    fwl_mult: float = 0.1
    """Focus-weighted loss multiplier"""


class FWLModel(NeuSFactoModel):
    """MonoSDF model

    Args:
        config: MonoSDF configuration to instantiate model
    """

    config: FWLModelConfig

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        focus_mask = batch["focus_mask"].to(self.device)
        loss_dict['focus_weighted_loss'] = get_focus_weighted_loss(outputs["rgb"], focus_mask)

        # fwl_mult is actually used on the original rgb loss 

        # dont use the original rgb loss
        loss_dict['rgb_loss'] = loss_dict['rgb_loss'] * self.config.fwl_mult

        return loss_dict
