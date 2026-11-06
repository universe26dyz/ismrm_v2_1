"""Aggregate independent case audit directories without selecting a winner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or np.ptp(left[valid]) == 0 or np.ptp(right[valid]) == 0:
        return None
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _metric_stats(values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    return {"count": int(values.size), "median": float(np.quantile(values, .5)) if values.size else None, "p90": float(np.quantile(values, .9)) if values.size else None, "p95": float(np.quantile(values, .95)) if values.size else None, "p99": float(np.quantile(values, .99)) if values.size else None, "mean": float(values.mean()) if values.size else None, "std": float(values.std()) if values.size else None}


def _read_pairs(directory: Path) -> list[dict[str, Any]]:
    path = directory / "same_world_pairs.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_reliability_audits(audit_dirs: list[str | Path], output_dir: str | Path) -> dict[str, Path]:
    directories = [Path(item).resolve() for item in audit_dirs]
    if not directories:
        raise ValueError("at least one audit output directory is required")
    if any(not (directory / "manifest.json").is_file() or not (directory / "summary.json").is_file() for directory in directories):
        raise FileNotFoundError("each audit directory must contain manifest.json and summary.json")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"aggregate output_dir must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifests = [_read_json(directory / "manifest.json") for directory in directories]
    pairs = [pair | {"audit_dir": str(directory)} for directory in directories for pair in _read_pairs(directory)]
    case_rows, stack_rows, timing_rows, discrepancy_rows, gate_distributions = [], [], [], [], []
    for directory, manifest in zip(directories, manifests):
        summary_payload = _read_json(directory / "summary.json")
        rows = [row for row in pairs if row["audit_dir"] == str(directory)]
        subject, case = manifest.get("subject"), manifest.get("resolved_case", manifest.get("requested_case"))
        for stack in ("2ch", "4ch"):
            subset = [row for row in rows if row["target_stack"] == stack]
            def vector(name: str) -> np.ndarray:
                return np.asarray([float(row[name]) for row in subset if row.get(name) not in ("", "None", None)], dtype=float)
            def aligned(left_name: str, right_name: str) -> tuple[np.ndarray, np.ndarray]:
                selected = [(float(row[left_name]), float(row[right_name])) for row in subset if row.get(left_name) not in ("", "None", None) and row.get(right_name) not in ("", "None", None)]
                return (np.asarray([item[0] for item in selected], dtype=float), np.asarray([item[1] for item in selected], dtype=float))
            metrics = {name: _metric_stats(vector(name)) for name in ("observed_l1_normalized", "observed_cosine_raw", "observed_cosine_normalized", "timing_only_l1_normalized", "timing_only_cosine")}
            gates_l1 = summary_payload.get("l1_gate_dry_run", {}).get(stack, {})
            gates_cos = summary_payload.get("cosine_gate_dry_run", {}).get(stack, {})
            for metric, gate in (("observed_l1_normalized", gates_l1), ("observed_cosine_raw", gates_cos)):
                gate_distributions.append({"subject": subject, "case_id": case, "target_stack": stack, "metric": metric, "gate_quantile": gate.get("gate_quantile"), "tau_q90": gate.get("tau_q90"), "tau_srpan_literal": gate.get("tau_srpan_literal"), "progress": gate.get("progress", {})})
            stack_rows.append({"subject": subject, "case_id": case, "target_stack": stack, "pair_count": len(subset), "reference_valid_count": sum(row.get("reference_valid") == "True" for row in subset), "reference_valid_fraction": (sum(row.get("reference_valid") == "True" for row in subset) / len(subset) if subset else None), "l1_gate_q90": gates_l1.get("tau_q90"), "l1_gate_tau_srpan_literal": gates_l1.get("tau_srpan_literal"), "cosine_gate_q90": gates_cos.get("tau_q90"), "cosine_gate_tau_srpan_literal": gates_cos.get("tau_srpan_literal"), **{f"{name}_{key}": value for name, payload in metrics.items() for key, value in payload.items()}})
            l1_timing, timing = aligned("observed_l1_normalized", "timing_abs_mean_ms"); l1_timing_l2, timing_l2 = aligned("observed_l1_normalized", "timing_l2_ms")
            cosine_timing, cosine_timing_abs = aligned("observed_cosine_normalized", "timing_abs_mean_ms"); cosine_timing_l2, timing_l2_cosine = aligned("observed_cosine_normalized", "timing_l2_ms")
            l1_to_timing_only, timing_only_l1 = aligned("observed_l1_normalized", "timing_only_l1_normalized"); cosine_to_timing_only, timing_only_cosine = aligned("observed_cosine_normalized", "timing_only_cosine")
            timing_rows.append({"subject": subject, "case_id": case, "target_stack": stack, "corr_l1_vs_timing_abs_mean": _corr(l1_timing, timing), "corr_l1_vs_timing_l2": _corr(l1_timing_l2, timing_l2), "corr_cosine_vs_timing_abs_mean": _corr(cosine_timing, cosine_timing_abs), "corr_cosine_vs_timing_l2": _corr(cosine_timing_l2, timing_l2_cosine), "corr_l1_vs_timing_only": _corr(l1_to_timing_only, timing_only_l1), "corr_cosine_vs_timing_only": _corr(cosine_to_timing_only, timing_only_cosine)})
            for metric in ("observed_l1_normalized", "observed_cosine_normalized"):
                discrepancy_rows.append({"subject": subject, "case_id": case, "target_stack": stack, "metric": metric, **_metric_stats(vector(metric))})
        case_rows.append({"subject": subject, "case_id": case, "status": manifest.get("status"), "pair_count": sum(row["subject"] == subject and row["case_id"] == case for row in rows), "reference_sampling_status": manifest.get("reference_sampling", "unknown"), "reference_valid_fraction": summary_payload.get("reference_valid_fraction")})
    between_case = {}
    for metric in ("observed_l1_normalized", "observed_cosine_normalized"):
        medians = []
        for row in case_rows:
            values = np.asarray([float(pair[metric]) for pair in pairs if pair["subject"] == row["subject"] and pair["case_id"] == row["case_id"] and pair.get(metric) not in ("", "None", None)], dtype=float)
            if values.size:
                medians.append(float(np.median(values)))
        medians_array = np.asarray(medians, dtype=float)
        between_case[metric] = {"case_count": int(medians_array.size), "case_median_cv": float(medians_array.std() / abs(medians_array.mean())) if medians_array.size > 1 and medians_array.mean() != 0 else None, "case_median_robust_spread": float(np.quantile(medians_array, .75) - np.quantile(medians_array, .25)) if medians_array.size else None}
    def stress(rows):
        return {metric: _metric_stats(np.asarray([float(pair[metric]) for pair in rows if pair.get(metric) not in ("", "None", None)], dtype=float)) for metric in ("observed_l1_normalized", "observed_cosine_normalized", "timing_only_l1_normalized", "timing_only_cosine")}
    dyl_l100 = [pair for pair in pairs if pair["subject"] == "DYL_2" and "L100" in str(pair["case_id"]).upper()]
    dyl_asymmetric = [pair for pair in pairs if pair["subject"] == "DYL_2" and "L100" not in str(pair["case_id"]).upper()]
    non_dyl_pairs = [pair for pair in pairs if pair["subject"] != "DYL_2"]
    heartbeat_stress = {"DYL_2_L100": stress(dyl_l100), "DYL_2_asymmetric_sparse": stress(dyl_asymmetric), "non_DYL_2": stress(non_dyl_pairs)}
    summary = {"schema": "ismrm_v2_2/reliability_audit_aggregate/v1", "audit_count": len(directories), "subjects": sorted({manifest.get("subject") for manifest in manifests}), "cases": case_rows, "metrics": {metric: _metric_stats(np.asarray([float(row[metric]) for row in pairs if row.get(metric) not in ("", "None", None)], dtype=float)) for metric in ("observed_l1_normalized", "observed_cosine_normalized")}, "between_case_dependence": between_case, "heartbeat_stress": heartbeat_stress, "gate_distributions": gate_distributions, "decision_states": ["L1_SUPPORTED", "COSINE_SUPPORTED", "INCONCLUSIVE_OR_TIMING_CONFOUNDED"], "decision": "INCONCLUSIVE_OR_TIMING_CONFOUNDED", "winner_selected": False, "interpretation": "Evidence only; no discrepancy is connected to training."}
    paths = {name: output / name for name in ("aggregate_summary.json", "case_table.csv", "stack_metric_table.csv", "timing_confounding_table.csv", "discrepancy_comparison.csv", "decision_support.json")}
    paths["aggregate_summary.json"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["decision_support.json"].write_text(json.dumps({"recommendation_states": summary["decision_states"], "recommendation": summary["decision"], "winner_selected": False, "evidence": summary["metrics"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _csv(paths["case_table.csv"], case_rows)
    _csv(paths["stack_metric_table.csv"], stack_rows)
    _csv(paths["timing_confounding_table.csv"], timing_rows)
    _csv(paths["discrepancy_comparison.csv"], discrepancy_rows)
    return paths


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", action="append", required=True, help="One completed single-case audit directory; repeat eight times.")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps({name: str(path) for name, path in aggregate_reliability_audits(args.audit_dir, args.output_dir).items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
