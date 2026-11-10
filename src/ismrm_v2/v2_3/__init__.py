"""v2.3-only joint fingerprint uncertainty infrastructure."""

from .hgnll import (
    HGNLLConfig,
    HGNLL_INITIAL_RAW_PARAMETER_BIAS,
    HGNLL_LOG_VARIANCE_INITIAL,
    HGNLL_SIGMA_INITIAL,
    HGNLL_SIGMA_MAX,
    HGNLL_VARIANCE_POLICY_VERSION,
    JointFingerprintVarianceHead,
    aggregate_log_variance,
    hgnll_objective,
)

__all__ = [
    "HGNLLConfig",
    "HGNLL_SIGMA_MAX",
    "HGNLL_SIGMA_INITIAL",
    "HGNLL_LOG_VARIANCE_INITIAL",
    "HGNLL_INITIAL_RAW_PARAMETER_BIAS",
    "HGNLL_VARIANCE_POLICY_VERSION",
    "JointFingerprintVarianceHead",
    "aggregate_log_variance",
    "hgnll_objective",
]
