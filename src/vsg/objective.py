"""Loss functions."""

import torch
import torch.distributions as td
from distribution_extension import BernoulliStraightThrough
from torch import Tensor


def likelihood(
    prediction: Tensor,
    target: Tensor,
    event_ndims: int,
    scale: float = 1.0,
) -> Tensor:
    """Compute the negative log-likelihood."""
    dist = td.Independent(td.Normal(prediction, scale), event_ndims)  # type: ignore[no-untyped-call]
    log_prob: Tensor = dist.log_prob(target)  # type: ignore[no-untyped-call]
    return -log_prob.mean()


def sparsity(
    update: BernoulliStraightThrough,
    gate_prob: float,
) -> Tensor:
    """Compute the sparsity loss."""
    if not isinstance(probs := update.probs, torch.Tensor):
        msg = f"Expected probs to be a Tensor, got {type(probs)}"
        raise TypeError(msg)
    probs = torch.ones_like(probs) * gate_prob
    kld = td.kl_divergence(
        q=update.independent(1),
        p=BernoulliStraightThrough(probs).independent(1),
    )
    return kld.mean()
