import torch
from torch.nn import functional as F
import utils.lorentz as L


def compute_stable_entailment_loss(
    parent: torch.Tensor,
    child: torch.Tensor,
    curv: torch.Tensor,
    min_radius: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    sin_aperture = (
        2 * min_radius / (torch.norm(parent, dim=-1) * torch.sqrt(curv) + eps)
    )

    clamped_sin_aperture = torch.clamp(sin_aperture, min=0.0, max=1.0 - eps)

    apers = torch.asin(clamped_sin_aperture)

    loss_diff = L.oxy_angle(parent, child, curv) - apers
    entailment_loss = F.softplus(loss_diff, beta=10).mean()

    return entailment_loss
