import torch
import torch.nn.functional as F
from torch import Tensor


def bradley_terry_loss(
    energy_preferred: Tensor,
    energy_dispreferred: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Bradley-Terry pairwise comparison loss.

    Encourages energy_preferred < energy_dispreferred.

    L = mean[ log(1 + exp((E_preferred - E_dispreferred) / tau)) ]

    When energy_preferred is correctly lower, loss ≈ 0.
    When energy_preferred is incorrectly higher, loss grows linearly.

    Parameters
    ----------
    energy_preferred : Tensor
        Energy for the preferred (matched / closer) complexes.
        Shape: [N] or [N, 1].
    energy_dispreferred : Tensor
        Energy for the dispreferred (swapped / farther) complexes.
        Shape: [N] or [N, 1].
    temperature : float
        Scaling temperature. Lower = sharper comparison.

    Returns
    -------
    Tensor
        Scalar loss (mean over the batch).
    """
    diff = (energy_preferred.flatten() - energy_dispreferred.flatten()) / temperature
    return torch.mean(F.softplus(diff))


def margin_ranking_loss(
    energy_close: Tensor,
    energy_far: Tensor,
    dist_close: Tensor,
    dist_far: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Margin ranking loss scaled by evolutionary distance gap.

    Enforces: E_far - E_close >= alpha * (d_far - d_close)

    The energy gap between two candidates should be at least proportional
    to their evolutionary distance gap from the anchor.

    L = mean[ log(1 + exp(alpha * (d_far - d_close) - (E_far - E_close))) ]

    Parameters
    ----------
    energy_close : Tensor
        Energy for the evolutionarily closer species. Shape: [N] or [N, 1].
    energy_far : Tensor
        Energy for the evolutionarily farther species. Shape: [N] or [N, 1].
    dist_close : Tensor
        Evolutionary distance to the closer species. Shape: [N] or [N, 1].
    dist_far : Tensor
        Evolutionary distance to the farther species. Shape: [N] or [N, 1].
    alpha : float
        Scaling factor for the distance-based margin.

    Returns
    -------
    Tensor
        Scalar loss (mean over the batch).
    """
    energy_gap = energy_far.flatten() - energy_close.flatten()
    dist_gap = dist_far.flatten() - dist_close.flatten()
    margin = alpha * dist_gap
    return torch.mean(F.softplus(margin - energy_gap))


def listwise_kl_loss(
    energies: Tensor,
    distances: Tensor,
    beta: float = 1.0,
    temperature: float = 1.0,
) -> Tensor:
    """Listwise KL divergence between distance- and energy-induced distributions.

    For a fixed anchor species s with K candidate partners:
      Target:  q(t | s) proportional to exp(-beta * d(s, t))
      Model:   p(t | s) proportional to exp(-E(s, t) / temperature)
      Loss:    KL(q || p)

    This trains the energy landscape so that its induced distribution
    over partners matches the evolutionary distance structure.

    Parameters
    ----------
    energies : Tensor
        Predicted energies for K candidates of one anchor.
        Shape: [K] or [K, 1].
    distances : Tensor
        Evolutionary distances d(s, t) for K candidates.
        Shape: [K] or [K, 1].
    beta : float
        Temperature for the target distribution. Higher beta =
        sharper preference for closer species.
    temperature : float
        Temperature for the model distribution. Higher temperature =
        softer energy landscape.

    Returns
    -------
    Tensor
        Scalar KL divergence loss.
    """
    target_logits = -beta * distances.flatten()
    model_logits = -energies.flatten() / temperature

    q = F.softmax(target_logits, dim=-1)
    log_p = F.log_softmax(model_logits, dim=-1)

    return F.kl_div(log_p, q, reduction="sum")


def evolution_loss(
    energies_preferred: Tensor,
    energies_dispreferred: Tensor,
    distances_preferred: Tensor = None,
    distances_dispreferred: Tensor = None,
    bt_weight: float = 1.0,
    margin_weight: float = 0.0,
    bt_temperature: float = 1.0,
    margin_alpha: float = 1.0,
) -> dict:
    """Combined evolution loss.

    Parameters
    ----------
    energies_preferred : Tensor
        Energies for preferred (matched / closer) pairs. Shape: [N].
    energies_dispreferred : Tensor
        Energies for dispreferred (swapped / farther) pairs. Shape: [N].
    distances_preferred : Tensor, optional
        Evolutionary distances for preferred pairs. Required when
        margin_weight > 0.
    distances_dispreferred : Tensor, optional
        Evolutionary distances for dispreferred pairs. Required when
        margin_weight > 0.
    bt_weight : float
        Weight for Bradley-Terry loss.
    margin_weight : float
        Weight for margin ranking loss.
    bt_temperature : float
        Temperature for BT comparison.
    margin_alpha : float
        Margin scaling factor for distance-proportional gaps.

    Returns
    -------
    dict
        'loss': total scalar loss
        'loss_breakdown': dict with 'bt_loss' and 'margin_loss'
    """
    bt_loss = bradley_terry_loss(
        energies_preferred, energies_dispreferred, bt_temperature
    )

    m_loss = torch.tensor(0.0, device=energies_preferred.device)
    if margin_weight > 0 and distances_preferred is not None:
        m_loss = margin_ranking_loss(
            energies_preferred,
            energies_dispreferred,
            distances_preferred,
            distances_dispreferred,
            margin_alpha,
        )

    total = bt_weight * bt_loss + margin_weight * m_loss

    return {
        "loss": total,
        "loss_breakdown": {
            "bt_loss": bt_loss,
            "margin_loss": m_loss,
        },
    }
