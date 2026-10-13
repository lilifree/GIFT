"""
Loss functions for three on-policy self-distillation methods.

- OPSD: Forward KL / Generalized JSD between teacher and student distributions
- SDPO: KL/JSD with top-k logit distillation and importance sampling
- RLSD: GRPO with evidence-ratio modulated token-level advantages
"""

import torch
import torch.nn.functional as F
from typing import Optional

ROLLOUT_IS_SAFETY_BOUND = 20.0


def _parse_rollout_is_threshold(threshold_spec):
    """Parse rollout IS threshold supporting TIS and IcePop-style bounds."""
    if isinstance(threshold_spec, bool):
        raise TypeError("rollout_is_threshold must be a float or threshold string, not a boolean.")

    if isinstance(threshold_spec, (int, float)):
        upper = float(threshold_spec)
        lower = None
    else:
        spec = str(threshold_spec).strip()
        if not spec:
            raise ValueError("rollout_is_threshold must not be empty.")
        if "_" in spec:
            lower_str, upper_str = spec.split("_", 1)
            lower = float(lower_str)
            upper = float(upper_str)
        else:
            lower = None
            upper = float(spec)

    if upper <= 0:
        raise ValueError(f"rollout_is_threshold upper bound must be positive, got {upper}.")
    if lower is not None:
        if lower <= 0:
            raise ValueError(f"rollout_is_threshold lower bound must be positive, got {lower}.")
        if lower > upper:
            raise ValueError("rollout_is_threshold lower bound must be <= upper bound.")

    return upper, lower


def compute_rollout_is_weights(
    old_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    mask: torch.Tensor,
    rollout_is: str = "token",
    rollout_is_threshold=2.0,
    rollout_is_batch_normalize: bool = False,
):
    """Compute truncated rollout-correction IS weights for the legacy trainer path."""
    rollout_is = str(rollout_is).strip().lower()
    if rollout_is not in {"token", "sequence"}:
        raise ValueError(f"Unsupported rollout_is={rollout_is!r}. Supported values: token, sequence.")

    upper, lower = _parse_rollout_is_threshold(rollout_is_threshold)
    log_ratio = old_log_probs - rollout_log_probs

    if rollout_is == "token":
        raw_weights = torch.exp(torch.clamp(log_ratio, min=-ROLLOUT_IS_SAFETY_BOUND, max=ROLLOUT_IS_SAFETY_BOUND))
    else:
        seq_log_ratio = (log_ratio * mask).sum(dim=-1, keepdim=True)
        seq_log_ratio = torch.clamp(seq_log_ratio, min=-ROLLOUT_IS_SAFETY_BOUND, max=ROLLOUT_IS_SAFETY_BOUND)
        raw_weights = torch.exp(seq_log_ratio).expand_as(log_ratio)

    raw_weights = raw_weights * mask
    if lower is None:
        rollout_is_weights = raw_weights.clamp(max=upper)
    else:
        keep = (raw_weights >= lower) & (raw_weights <= upper)
        rollout_is_weights = torch.where(keep, raw_weights, torch.zeros_like(raw_weights))

    metrics = {}
    valid = mask.bool()
    if valid.any():
        valid_weights = rollout_is_weights[valid]
        valid_raw = raw_weights[valid]
        metrics = {
            "rollout_is_mean": valid_weights.mean().item(),
            "rollout_is_std": valid_weights.std().item() if valid_weights.numel() > 1 else 0.0,
            "rollout_is_min": valid_weights.min().item(),
            "rollout_is_max": valid_weights.max().item(),
            "rollout_is_fraction_high": (valid_raw > upper).float().mean().item(),
            "rollout_is_fraction_low": (
                (valid_raw < lower).float().mean().item() if lower is not None else 0.0
            ),
        }

        if rollout_is_batch_normalize:
            if rollout_is == "token":
                mean_weight = valid_weights.mean()
            else:
                seq_mask = mask.sum(dim=-1) > 0
                seq_weights = (rollout_is_weights * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
                mean_weight = seq_weights[seq_mask].mean() if seq_mask.any() else None

            if mean_weight is not None and mean_weight.item() > 1e-8:
                rollout_is_weights = rollout_is_weights / mean_weight
                metrics["rollout_is_batch_norm_factor"] = mean_weight.item()

    return rollout_is_weights.detach(), metrics


def generalized_jsd_loss(
    student_logits: torch.Tensor,   # (B, T, V)
    teacher_logits: torch.Tensor,   # (B, T, V)
    mask: torch.Tensor,             # (B, T)
    beta: float = 0.0,             # 0=forward KL, 1=reverse KL, 0.5=symmetric JSD
    temperature: float = 1.0,
    jsd_token_clip: float = 0.0,    # per-token KL clipping (OPSD uses 0.05)
    top_k: int = 0,                 # top-k restriction on teacher logits (SDPO)
) -> torch.Tensor:
    """
    Generalized JSD loss (OPSD & SDPO).

    beta=0: Forward KL = KL(teacher || student)  -- OPSD default
    beta=1: Reverse KL = KL(student || teacher)
    0<beta<1: JSD_beta = beta * KL(teacher || M) + (1-beta) * KL(student || M)

    Args:
        student_logits: Student model logits (B, T, V)
        teacher_logits: Teacher model logits (B, T, V)
        mask: Valid token mask (B, T)
        beta: JSD interpolation weight
        temperature: Temperature for softmax
        jsd_token_clip: Max per-token divergence (0 = no clip)
        top_k: If > 0, restrict to top-k teacher tokens
    """
    # Apply temperature
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    if top_k > 0:
        # SDPO-style: restrict to top-k teacher tokens
        topk_values, topk_indices = torch.topk(teacher_logits, top_k, dim=-1)
        teacher_log_probs = F.log_softmax(topk_values, dim=-1)
        student_topk = torch.gather(student_logits, -1, topk_indices)
        student_log_probs = F.log_softmax(student_topk, dim=-1)
    else:
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

    if beta == 0.0:
        # Forward KL: KL(teacher || student) = sum teacher * (log teacher - log student)
        per_token = F.kl_div(
            student_log_probs, teacher_log_probs,
            log_target=True, reduction="none"
        ).sum(-1)  # (B, T)
    elif beta == 1.0:
        # Reverse KL: KL(student || teacher)
        per_token = F.kl_div(
            teacher_log_probs, student_log_probs,
            log_target=True, reduction="none"
        ).sum(-1)
    else:
        # Generalized JSD
        log_beta = torch.tensor(beta, device=student_logits.device).log()
        log_1_minus_beta = torch.tensor(1 - beta, device=student_logits.device).log()

        mixture_log_probs = torch.logsumexp(
            torch.stack([
                student_log_probs + log_1_minus_beta,
                teacher_log_probs + log_beta,
            ], dim=0),
            dim=0,
        )

        kl_teacher = F.kl_div(
            mixture_log_probs, teacher_log_probs,
            log_target=True, reduction="none"
        ).sum(-1)
        kl_student = F.kl_div(
            mixture_log_probs, student_log_probs,
            log_target=True, reduction="none"
        ).sum(-1)

        per_token = beta * kl_teacher + (1 - beta) * kl_student

    # Per-token clipping (OPSD stabilization)
    if jsd_token_clip > 0:
        per_token = per_token.clamp(max=jsd_token_clip)

    # Apply mask and reduce
    masked_loss = per_token * mask
    total_tokens = mask.sum().clamp(min=1)
    loss = masked_loss.sum() / total_tokens

    return loss


def sdpo_loss(
    student_logits: torch.Tensor,   # (B, T, V)
    teacher_logits: torch.Tensor,   # (B, T, V)
    old_log_probs: torch.Tensor,    # (B, T) -- log probs from rollout policy
    mask: torch.Tensor,             # (B, T)
    distillation_mask: torch.Tensor,  # (B,) -- which samples have teacher context
    alpha: float = 0.5,
    top_k: int = 100,
    is_clip: float = 2.0,          # importance sampling clip
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    SDPO loss: KL/JSD distillation with importance sampling correction.

    Only applied to samples with successful demonstrations (distillation_mask=1).
    Uses top-k logit restriction and importance sampling ratio clipping.

    Args:
        student_logits: Current policy logits
        teacher_logits: Teacher (EMA model + reprompt context) logits
        old_log_probs: Log probs from the rollout (for IS correction)
        mask: Valid token mask
        distillation_mask: Which samples have a teacher signal (B,)
        alpha: JSD interpolation (0=forward KL, 0.5=JSD, 1=reverse KL)
        top_k: Top-k logit restriction
        is_clip: Importance sampling ratio clip
        temperature: Temperature
    """
    if distillation_mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    # Filter to samples with teacher context
    idx = distillation_mask.bool()
    student_logits = student_logits[idx]
    teacher_logits = teacher_logits[idx]
    old_lp = old_log_probs[idx]
    mask = mask[idx]

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    if top_k > 0:
        topk_values, topk_indices = torch.topk(teacher_logits, top_k, dim=-1)
        teacher_log_probs = F.log_softmax(topk_values, dim=-1)

        student_topk = torch.gather(student_logits, -1, topk_indices)
        student_log_probs = F.log_softmax(student_topk, dim=-1)

        # Add tail bucket
        teacher_full_lp = F.log_softmax(teacher_logits, dim=-1)
        student_full_lp = F.log_softmax(student_logits, dim=-1)

        teacher_topk_sum = torch.logsumexp(
            torch.gather(teacher_full_lp, -1, topk_indices), dim=-1, keepdim=True
        )
        student_topk_sum = torch.logsumexp(
            torch.gather(student_full_lp, -1, topk_indices), dim=-1, keepdim=True
        )

        teacher_tail = torch.log1p(-teacher_topk_sum.exp().clamp(max=1 - 1e-7))
        student_tail = torch.log1p(-student_topk_sum.exp().clamp(max=1 - 1e-7))

        teacher_log_probs = torch.cat([teacher_log_probs, teacher_tail], dim=-1)
        student_log_probs = torch.cat([student_log_probs, student_tail], dim=-1)
    else:
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

    # Compute KL/JSD
    if alpha == 0.0:
        per_token_kl = F.kl_div(
            student_log_probs, teacher_log_probs,
            log_target=True, reduction="none"
        ).sum(-1)
    elif alpha == 1.0:
        per_token_kl = F.kl_div(
            teacher_log_probs, student_log_probs,
            log_target=True, reduction="none"
        ).sum(-1)
    else:
        log_alpha = torch.tensor(alpha, device=student_logits.device).log()
        log_1_minus_alpha = torch.tensor(1 - alpha, device=student_logits.device).log()
        mixture = torch.logsumexp(
            torch.stack([
                student_log_probs + log_1_minus_alpha,
                teacher_log_probs + log_alpha,
            ], dim=0),
            dim=0,
        )
        kl_t = F.kl_div(mixture, teacher_log_probs, log_target=True, reduction="none").sum(-1)
        kl_s = F.kl_div(mixture, student_log_probs, log_target=True, reduction="none").sum(-1)
        per_token_kl = alpha * kl_t + (1 - alpha) * kl_s

    # Importance sampling correction
    # Current student log probs for sampled tokens
    current_lp = F.log_softmax(student_logits, dim=-1)
    # Get log prob of the tokens that were actually generated
    # old_lp is already per-token log probs from rollout
    # Approximate IS ratio
    neg_approx_kl = (current_lp.max(-1).values - old_lp).detach()
    ratio = neg_approx_kl.clamp(-20, 20).exp().clamp(max=is_clip)

    per_token_loss = per_token_kl * ratio
    masked_loss = per_token_loss * mask
    total_tokens = mask.sum().clamp(min=1)

    return masked_loss.sum() / total_tokens


def rlsd_loss(
    student_logits: torch.Tensor,   # (B, T, V) -- current policy
    teacher_logits: torch.Tensor,   # (B, T, V) -- teacher (conditioned on answer)
    old_log_probs: torch.Tensor,    # (B, T) -- log probs from old policy
    response_ids: torch.Tensor,     # (B, T) -- token ids of responses
    mask: torch.Tensor,             # (B, T)
    advantages: torch.Tensor,       # (B,) -- sequence-level GRPO advantages
    epsilon: float = 0.2,          # PPO/GRPO clip
    epsilon_w: float = 0.2,        # evidence ratio clip
    lam: float = 0.5,             # mixing coefficient for reweighting
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    RLSD loss: GRPO with evidence-ratio modulated token-level advantages.

    The environment reward determines direction (via advantage sign),
    while teacher/student evidence ratio modulates magnitude.

    Args:
        student_logits: Current policy logits
        teacher_logits: Teacher logits (model conditioned on ground-truth)
        old_log_probs: Log probs from old policy
        response_ids: Token IDs of generated responses
        mask: Valid token mask
        advantages: Sequence-level advantages from GRPO (B,)
        epsilon: GRPO clipping parameter
        epsilon_w: Evidence ratio clipping bound
        lam: Mixing coefficient (0=uniform, 1=full reweighting)
    """
    # Current policy log probs for generated tokens
    current_log_probs = F.log_softmax(student_logits, dim=-1)
    current_lp = torch.gather(current_log_probs, -1, response_ids.unsqueeze(-1)).squeeze(-1)

    # Teacher log probs for generated tokens (no grad)
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_lp = torch.gather(teacher_log_probs, -1, response_ids.unsqueeze(-1)).squeeze(-1)

        # Student log probs (from current forward, detached for ratio computation)
        student_lp_detach = current_lp.detach()

        # Privileged information gain: delta_t = log P_T(y_t) - log P_S(y_t)
        delta_t = teacher_lp - student_lp_detach  # (B, T)

        # Direction-aware reweighting: w_t = exp(sign(A) * delta_t)
        adv_sign = advantages.sign().unsqueeze(-1)  # (B, 1)
        w_t = (adv_sign * delta_t).exp()  # (B, T)

        # Clip w_t
        w_t = w_t.clamp(1 - epsilon_w, 1 + epsilon_w)

        # Modified per-token advantage: A_hat_t = A * ((1-lam) + lam * w_t)
        adv_expanded = advantages.unsqueeze(-1)  # (B, 1)
        modified_advantages = adv_expanded * ((1 - lam) + lam * w_t)  # (B, T)

    # Standard GRPO policy ratio
    log_ratio = current_lp - old_log_probs
    ratio = log_ratio.exp()

    # Clipped surrogate objective
    surr1 = ratio * modified_advantages
    surr2 = ratio.clamp(1 - epsilon, 1 + epsilon) * modified_advantages
    per_token_loss = -torch.min(surr1, surr2)
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    # Mask and reduce
    masked_loss = per_token_loss * mask
    total_tokens = mask.sum().clamp(min=1)

    return masked_loss.sum() / total_tokens


def compute_grpo_advantages(
    rewards: torch.Tensor,  # (B,) or (B, G) for grouped
    group_size: int = 1,
) -> torch.Tensor:
    """
    Compute GRPO group-relative advantages.

    For each prompt group, normalize rewards to zero mean unit variance.
    """
    if rewards.dim() == 1 and group_size > 1:
        B = rewards.shape[0]
        assert B % group_size == 0
        rewards = rewards.view(-1, group_size)

    if rewards.dim() == 2:
        # Group-level normalization
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
        advantages = (rewards - mean) / std
        return advantages.view(-1)
    else:
        # Global normalization
        mean = rewards.mean()
        std = rewards.std().clamp(min=1e-8)
        return (rewards - mean) / std


def compute_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute average token-level entropy from logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(-1)  # (B, T)
    masked_entropy = entropy * mask
    return masked_entropy.sum() / mask.sum().clamp(min=1)
