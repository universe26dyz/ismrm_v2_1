"""Joint-fingerprint heteroscedastic Gaussian negative log likelihood.

This module is deliberately v2.3-specific.  It owns the new ten-output
variance branch and its numerical policy; the v2.2 objectives and reliability
cache remain untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


HGNLL_DISTRIBUTION = "gaussian"
HGNLL_OUTPUT = "per_weight_log_variance"
HGNLL_AGGREGATION = "log_mean_exp_variance"
HGNLL_LOG_VARIANCE_MIN = -10.0
HGNLL_SIGMA_MAX = 1.0
HGNLL_LOG_VARIANCE_MAX = math.log(HGNLL_SIGMA_MAX**2)
HGNLL_SIGMA_INITIAL = 0.5
HGNLL_LOG_VARIANCE_INITIAL = math.log(HGNLL_SIGMA_INITIAL**2)
HGNLL_VARIANCE_POLICY_VERSION = "multimap_hgnll_smooth_sigma1_v1"
HGNLL_INITIALIZATION = "inverse_sigmoid_of_sigma_initial"


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _initial_raw_parameter_bias() -> float:
    probability = (HGNLL_LOG_VARIANCE_INITIAL - HGNLL_LOG_VARIANCE_MIN) / (HGNLL_LOG_VARIANCE_MAX - HGNLL_LOG_VARIANCE_MIN)
    return _logit(probability)


HGNLL_INITIAL_RAW_PARAMETER_BIAS = _initial_raw_parameter_bias()


@dataclass(frozen=True)
class HGNLLConfig:
    """Explicit route-scoped HGNLL policy and provenance schema."""

    enabled: bool = False
    distribution: str = HGNLL_DISTRIBUTION
    output: str = HGNLL_OUTPUT
    weights_per_anchor: int = 10
    psf_variance_aggregation: str = HGNLL_AGGREGATION
    log_variance_min: float = HGNLL_LOG_VARIANCE_MIN
    log_variance_min_reason: str = "project_numerical_safety_floor"
    sigma_max: float = HGNLL_SIGMA_MAX
    sigma_max_reason: str = "pre_existing_v2_2_mse_residual_scale_audit"
    log_variance_max: float = HGNLL_LOG_VARIANCE_MAX
    sigma_initial: float = HGNLL_SIGMA_INITIAL
    log_variance_initial: float = HGNLL_LOG_VARIANCE_INITIAL
    raw_parameter_initial_bias: float = HGNLL_INITIAL_RAW_PARAMETER_BIAS
    variance_policy_version: str = HGNLL_VARIANCE_POLICY_VERSION
    initialization: str = HGNLL_INITIALIZATION
    loss_prefactor: float = 0.5
    hidden_features: int = 32
    variance_learning_rate_key: str = "variance"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None, *, required: bool = False) -> "HGNLLConfig":
        if value is None:
            if required:
                raise ValueError("HGNLL routes require an explicit heteroscedastic configuration")
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("heteroscedastic must be a mapping")
        payload = dict(value)
        enabled = bool(payload.get("enabled", False))
        if required and not enabled:
            raise ValueError("HGNLL routes require heteroscedastic.enabled=true")
        # Do not accept arbitrary policy strings or silently substitute them.
        allowed = {
            "enabled", "distribution", "output", "weights_per_anchor",
            "psf_variance_aggregation", "log_variance_min", "sigma_max",
            "log_variance_min_reason", "sigma_max_reason", "log_variance_max", "sigma_initial", "log_variance_initial",
            "raw_parameter_initial_bias", "variance_policy_version", "initialization",
            "loss_prefactor", "hidden_features", "variance_learning_rate_key",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown HGNLL configuration fields: {unknown}")
        cfg = cls(**payload)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.distribution != HGNLL_DISTRIBUTION:
            raise ValueError("HGNLL distribution is frozen to gaussian")
        if self.output != HGNLL_OUTPUT:
            raise ValueError("HGNLL output is frozen to per_weight_log_variance")
        if self.weights_per_anchor != 10:
            raise ValueError("HGNLL requires exactly ten weights per anchor")
        if self.psf_variance_aggregation != HGNLL_AGGREGATION:
            raise ValueError("HGNLL PSF variance aggregation is frozen to log_mean_exp_variance")
        if not math.isfinite(self.log_variance_min) or not math.isfinite(self.log_variance_max) or self.log_variance_min >= self.log_variance_max:
            raise ValueError("HGNLL log-variance bounds must be finite and ordered")
        if self.log_variance_min != HGNLL_LOG_VARIANCE_MIN:
            raise ValueError("HGNLL log_variance_min is frozen to -10")
        if self.log_variance_min_reason != "project_numerical_safety_floor":
            raise ValueError("HGNLL lower bound must be documented as the project numerical-safety floor")
        if not math.isclose(self.sigma_max, HGNLL_SIGMA_MAX, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("HGNLL sigma_max is frozen to 1.0")
        if self.sigma_max_reason != "pre_existing_v2_2_mse_residual_scale_audit":
            raise ValueError("HGNLL sigma_max must be documented as the pre-existing v2.2 MSE residual-scale audit")
        if not math.isclose(self.log_variance_max, HGNLL_LOG_VARIANCE_MAX, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("HGNLL log_variance_max must equal log(1.0**2)")
        if not math.isclose(self.sigma_initial, HGNLL_SIGMA_INITIAL, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("HGNLL sigma_initial is frozen to 0.5")
        if not math.isclose(self.log_variance_initial, HGNLL_LOG_VARIANCE_INITIAL, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("HGNLL log_variance_initial must equal log(0.5**2)")
        expected_bias = _initial_raw_parameter_bias()
        if not math.isclose(self.raw_parameter_initial_bias, expected_bias, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("HGNLL raw_parameter_initial_bias disagrees with the inverse-sigmoid initialization")
        if self.variance_policy_version != HGNLL_VARIANCE_POLICY_VERSION:
            raise ValueError("unknown HGNLL variance policy version")
        if self.initialization != HGNLL_INITIALIZATION:
            raise ValueError("HGNLL initialization must use inverse-sigmoid sigma_initial")
        if self.loss_prefactor != 0.5:
            raise ValueError("HGNLL loss prefactor is frozen to 0.5")
        if self.hidden_features < 1 or not self.variance_learning_rate_key:
            raise ValueError("HGNLL head and optimizer settings are invalid")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "enabled": bool(self.enabled),
            "distribution": self.distribution,
            "output": self.output,
            "weights_per_anchor": int(self.weights_per_anchor),
            "psf_variance_aggregation": self.psf_variance_aggregation,
            "log_variance_min": float(self.log_variance_min),
            "log_variance_min_reason": self.log_variance_min_reason,
            "sigma_max": float(self.sigma_max),
            "sigma_max_reason": self.sigma_max_reason,
            "log_variance_max": float(self.log_variance_max),
            "sigma_initial": float(self.sigma_initial),
            "log_variance_initial": float(self.log_variance_initial),
            "raw_parameter_initial_bias": float(self.raw_parameter_initial_bias),
            "variance_policy_version": self.variance_policy_version,
            "initialization": self.initialization,
            "loss_prefactor": float(self.loss_prefactor),
            "hidden_features": int(self.hidden_features),
            "variance_learning_rate_key": self.variance_learning_rate_key,
        }


class JointFingerprintVarianceHead(nn.Module):
    """Predict ten unconstrained parameters and map them smoothly to log variance."""

    def __init__(self, input_features: int, config: HGNLLConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config if isinstance(config, HGNLLConfig) else HGNLLConfig.from_mapping(config, required=True)
        self.config.validate()
        if input_features < 1:
            raise ValueError("variance head input_features must be positive")
        self.network = nn.Sequential(
            nn.Linear(int(input_features), self.config.hidden_features),
            nn.SiLU(),
            nn.Linear(self.config.hidden_features, self.config.weights_per_anchor),
        )
        # A zero final weight makes the initial output deterministic.  The
        # inverse-sigmoid bias places the bounded value exactly at sigma=0.5.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, self.config.raw_parameter_initial_bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError("HGNLL variance head input must be [N,features]")
        if latent.shape[1] != self.network[0].in_features:
            raise ValueError("HGNLL variance head input feature count disagrees with the INR")
        return self.network(latent)

    def bounded_log_variance_from_raw(self, raw_variance_parameter: torch.Tensor) -> torch.Tensor:
        """Map unconstrained parameters smoothly into the configured interval."""

        return self.config.log_variance_min + (self.config.log_variance_max - self.config.log_variance_min) * torch.sigmoid(raw_variance_parameter)

    def bounded_log_variance(self, latent: torch.Tensor) -> torch.Tensor:
        return self.bounded_log_variance_from_raw(self(latent))


def attach_hgnll_head(model: nn.Module, config: HGNLLConfig | Mapping[str, Any]) -> JointFingerprintVarianceHead:
    """Register the v2.3 head on a reconstruction model, once, and return it."""

    cfg = config if isinstance(config, HGNLLConfig) else HGNLLConfig.from_mapping(config, required=True)
    if not cfg.enabled:
        raise ValueError("cannot attach a disabled HGNLL configuration")
    if getattr(model, "joint_variance_head", None) is not None:
        raise ValueError("model already has a joint HGNLL variance head")
    inr = getattr(model, "inr", None)
    latent_features = getattr(getattr(inr, "config", None), "latent_features", None)
    if latent_features is None:
        raise ValueError("HGNLL requires an INR exposing config.latent_features")
    head = JointFingerprintVarianceHead(int(latent_features), cfg).to(next(model.parameters()).device)
    model.joint_variance_head = head
    return head


def aggregate_log_variance(logvar_psf: torch.Tensor, *, psf_dim: int = 1) -> torch.Tensor:
    """Aggregate per-PSF log variance as log(mean(exp(logvar)))."""

    if logvar_psf.ndim != 3 or logvar_psf.shape[-1] != 10:
        raise ValueError("logvar_psf must have shape [B,K,10] (or a specified PSF dimension)")
    k = int(logvar_psf.shape[psf_dim])
    if k < 1:
        raise ValueError("PSF dimension must be non-empty")
    return torch.logsumexp(logvar_psf, dim=psf_dim) - math.log(k)


def _stack_weights(stack_idx: torch.Tensor, stack_weights: Mapping[int, float], *, dtype: torch.dtype) -> torch.Tensor:
    values = torch.full(stack_idx.shape, float("nan"), dtype=dtype, device=stack_idx.device)
    for stack, weight in stack_weights.items():
        values[stack_idx == int(stack)] = float(weight)
    if not torch.isfinite(values).all():
        raise ValueError("HGNLL stack weights do not cover every sampled anchor")
    return values


def hgnll_objective(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    log_variance: torch.Tensor,
    stack_idx: torch.Tensor,
    stack_weights: Mapping[int, float],
    *,
    reliability_multiplier: torch.Tensor | None = None,
    stack_balance_enabled: bool = True,
    raw_variance_parameter: torch.Tensor | None = None,
    config: HGNLLConfig | Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return complete-anchor HGNLL and detached diagnostics.

    The residual is reduced across the ten weights first, then stack balance
    and (when supplied) the Stage-B reliability multiplier act on that one
    complete-anchor scalar.  ``joint_signal_mse`` is the historical
    stack-balanced, reliability-independent MSE used for B1 comparison;
    ``raw_signal_mse`` is the completely unweighted diagnostic.  The two
    logged HGNLL terms are the final weighted decomposition of ``hgnll``.
    """

    cfg = config if isinstance(config, HGNLLConfig) else HGNLLConfig.from_mapping(config or {"enabled": True}, required=True)
    cfg.validate()
    if prediction.shape != observed.shape or prediction.ndim != 2 or prediction.shape[1] != 10 or log_variance.shape != prediction.shape:
        raise ValueError("HGNLL prediction, observation, and log_variance must be matching [B,10] tensors")
    stacks = torch.as_tensor(stack_idx, dtype=torch.long, device=prediction.device)
    if stacks.shape != prediction.shape[:1]:
        raise ValueError("HGNLL stack_idx must be [B]")
    lv = log_variance
    if torch.any(lv < cfg.log_variance_min) or torch.any(lv > cfg.log_variance_max):
        raise ValueError("bounded HGNLL log variance lies outside the configured interval")
    residual = prediction - observed
    per_weight_residual = cfg.loss_prefactor * torch.exp(-lv) * residual.pow(2)
    per_weight_logvar = cfg.loss_prefactor * lv
    per_anchor_residual = per_weight_residual.mean(dim=1)
    per_anchor_logvar = per_weight_logvar.mean(dim=1)
    weights = _stack_weights(stacks, stack_weights, dtype=prediction.dtype) if stack_balance_enabled else torch.ones_like(per_anchor_residual)
    multiplier = torch.ones_like(per_anchor_residual)
    if reliability_multiplier is not None:
        multiplier = torch.as_tensor(reliability_multiplier, dtype=prediction.dtype, device=prediction.device).detach()
        if multiplier.shape != per_anchor_residual.shape:
            raise ValueError("HGNLL reliability_multiplier must be [B]")
    # Keep the baseline MSE field independent of HGNLL/reliability while
    # retaining the historical stack-balanced semantics for B1 routes.
    raw_signal_mse = residual.pow(2).mean()
    joint_signal_mse = (weights * residual.pow(2).mean(dim=1)).mean()
    weighted_residual = (weights * multiplier * per_anchor_residual).mean()
    weighted_logvar = (weights * multiplier * per_anchor_logvar).mean()
    total = weighted_residual + weighted_logvar
    raw_parameter = log_variance.detach() if raw_variance_parameter is None else torch.as_tensor(raw_variance_parameter, device=log_variance.device).detach()
    if raw_parameter.shape != log_variance.shape and not (raw_parameter.ndim == 3 and raw_parameter.shape[0] == log_variance.shape[0] and raw_parameter.shape[-1] == 10):
        raise ValueError("raw_variance_parameter must be [B,10] or [B,K,10]")
    raw_cpu = raw_parameter.reshape(-1).cpu()
    bounded_cpu = lv.detach().reshape(-1).cpu()
    sigma_cpu = torch.exp(0.5 * bounded_cpu)
    sigma_matrix_cpu = torch.exp(0.5 * lv.detach()).cpu()
    sigma_max = float(cfg.sigma_max)
    lower_proximity = torch.abs(lv.detach() - cfg.log_variance_min) <= 0.01
    diagnostics: dict[str, Any] = {
        "hgnll": total,
        "joint_signal_mse": joint_signal_mse,
        "raw_signal_mse": raw_signal_mse,
        "data_loss": total,
        "hgnll_residual_term": weighted_residual,
        "hgnll_log_variance_term": weighted_logvar,
        "raw_variance_parameter_min": float(raw_cpu.min()),
        "raw_variance_parameter_max": float(raw_cpu.max()),
        "raw_variance_parameter_mean": float(raw_cpu.mean()),
        "bounded_logvar_min": float(bounded_cpu.min()),
        "bounded_logvar_max": float(bounded_cpu.max()),
        "bounded_logvar_mean": float(bounded_cpu.mean()),
        "sigma_mean": float(sigma_cpu.mean()),
        "sigma_p10": float(torch.quantile(sigma_cpu, 0.10)),
        "sigma_p50": float(torch.quantile(sigma_cpu, 0.50)),
        "sigma_p90": float(torch.quantile(sigma_cpu, 0.90)),
        "sigma_mean_by_weight": {str(index): float(value) for index, value in enumerate(sigma_matrix_cpu.mean(dim=0))},
        "sigma_p10_by_weight": {str(index): float(value) for index, value in enumerate(torch.quantile(sigma_matrix_cpu, 0.10, dim=0))},
        "sigma_p50_by_weight": {str(index): float(value) for index, value in enumerate(torch.quantile(sigma_matrix_cpu, 0.50, dim=0))},
        "sigma_p90_by_weight": {str(index): float(value) for index, value in enumerate(torch.quantile(sigma_matrix_cpu, 0.90, dim=0))},
        "fraction_sigma_gt_0_95_sigma_max": float((sigma_cpu > 0.95 * sigma_max).float().mean()),
        "fraction_sigma_gt_0_99_sigma_max": float((sigma_cpu > 0.99 * sigma_max).float().mean()),
        "fraction_logvar_within_0_01_of_lower_bound": float(lower_proximity.float().mean().detach().cpu()),
        "hgnll_reliability_multiplier": multiplier,
        "total_data_loss": total,
    }
    return total, diagnostics


# Descriptive aliases make the policy easy to discover without duplicating math.
hgnll_loss = hgnll_objective
log_mean_exp_variance = aggregate_log_variance


__all__ = [
    "HGNLLConfig", "JointFingerprintVarianceHead", "attach_hgnll_head", "aggregate_log_variance",
    "hgnll_objective", "hgnll_loss", "log_mean_exp_variance",
    "HGNLL_LOG_VARIANCE_MIN", "HGNLL_LOG_VARIANCE_MAX", "HGNLL_SIGMA_MAX",
    "HGNLL_SIGMA_INITIAL", "HGNLL_LOG_VARIANCE_INITIAL", "HGNLL_INITIAL_RAW_PARAMETER_BIAS",
    "HGNLL_VARIANCE_POLICY_VERSION",
]
