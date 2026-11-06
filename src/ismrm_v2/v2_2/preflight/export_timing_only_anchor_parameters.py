"""Export frozen-geometry, case-specific T1/T2/B1 parameters for timing-only audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cross_view_reliability_audit import _dataset_arrays, _joint_anchor_order_sha256, _sha256_file, load_geometry_checkpoint
from .same_world_reliability_audit import _manifest_active_inputs
from ismrm_v2.proposed.training.training_space import TrainingSpace
from ismrm_v2.proposed.trad_quantitative_inr import QuantitativeINR, QuantitativeINRConfig

APPROVED_DECODER_SHA256 = "338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b"


def _tensor_hash(value: Any) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def load_frozen_quantitative_state_for_timing_audit(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    fallback_b6_model: str | Path | None = None,
    current_space: TrainingSpace | None = None,
) -> tuple[torch.nn.Module, TrainingSpace, dict[str, Any]]:
    """Load only the frozen quantitative INR; never construct the signal decoder."""
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen quantitative checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("frozen quantitative checkpoint must contain a mapping")
    required = {"model_state", "training_space", "group_axisangle_post_stack_init_physical", "group_resolution_xyz_mm"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"frozen quantitative checkpoint lacks required fields: {sorted(missing)}")
    state = checkpoint["model_state"]
    if not isinstance(state, dict):
        raise ValueError("frozen quantitative checkpoint model_state must be a mapping")
    inr_state = {key.removeprefix("inr."): value for key, value in state.items() if key.startswith("inr.")}
    if not inr_state:
        raise ValueError("frozen quantitative checkpoint has no inr.* model state")
    if "intensity_scale" not in state:
        raise ValueError("frozen quantitative checkpoint lacks model_state.intensity_scale")
    resolved = checkpoint.get("resolved_config")
    fallback_used = False
    if not isinstance(resolved, dict):
        if fallback_b6_model is None:
            raise ValueError("checkpoint lacks resolved_config and no B6 fallback was supplied")
        from ismrm_v2.proposed.q001_config import load_b6_resolved_config
        resolved = load_b6_resolved_config(fallback_b6_model)
        fallback_used = True
    training = resolved.get("training", {})
    inr_config = resolved.get("inr")
    if not isinstance(training, dict) or not isinstance(inr_config, dict):
        raise ValueError("checkpoint quantitative config lacks training/inr mappings")
    space = TrainingSpace.from_state_dict(checkpoint["training_space"])
    required_inr = ("coarsest_resolution_mm", "finest_resolution_mm", "level_scale", "n_features_per_level", "log2_hashmap_size", "latent_features", "width", "depth")
    if any(key not in inr_config for key in required_inr):
        raise ValueError("checkpoint quantitative config lacks architecture-critical INR fields")
    model = QuantitativeINR(space.bbox_train, QuantitativeINRConfig(**{key: inr_config[key] for key in required_inr}), spatial_scaling=space.spatial_scaling).to(device)
    model.load_state_dict(inr_state, strict=True)
    scale = torch.as_tensor(state["intensity_scale"], dtype=torch.float64)
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
        raise ValueError("frozen checkpoint intensity_scale must be one finite positive scalar")
    normalization = checkpoint.get("intensity_normalization")
    normalization_scale = None
    if isinstance(normalization, dict) and normalization.get("scale") is not None:
        normalization_scale = float(normalization["scale"])
        if not np.isclose(scale.item(), normalization_scale, atol=1e-6, rtol=1e-6):
            raise ValueError("model_state.intensity_scale disagrees with intensity_normalization.scale")
    model.register_buffer("intensity_scale", scale.to(dtype=torch.float32, device=device))
    if current_space is not None:
        comparisons = {
            "spatial_scaling": (float(space.spatial_scaling), float(current_space.spatial_scaling)),
            "center_ras_mm": (space.center_ras_mm, current_space.center_ras_mm),
            "bbox_train": (space.bbox_train, current_space.bbox_train),
            "group_axisangle_init_physical": (space.group_axisangle_init_physical, current_space.group_axisangle_init_physical),
            "group_axisangle_init_train": (space.group_axisangle_init_train, current_space.group_axisangle_init_train),
            "group_resolution_xyz_mm": (space.group_resolution_xyz_mm, current_space.group_resolution_xyz_mm),
        }
        differences = {}
        for name, (left, right) in comparisons.items():
            differences[name] = float(torch.max(torch.abs(torch.as_tensor(left, dtype=torch.float64) - torch.as_tensor(right, dtype=torch.float64))))
            if differences[name] > 1e-5:
                raise ValueError(f"checkpoint training_space {name} mismatch: {differences[name]:g} > 1e-5")
    provenance = {
        "checkpoint_path": str(path), "checkpoint_sha256": _sha256_file(path),
        "checkpoint_training_space_sha256": hashlib.sha256(json.dumps({key: (value.tolist() if isinstance(value, torch.Tensor) else value) for key, value in space.state_dict().items()}, sort_keys=True, default=str).encode()).hexdigest(),
        "checkpoint_training_space_compatibility": True,
        "training_space_max_abs_diff": differences if current_space is not None else {},
        "intensity_scale": float(scale.item()), "intensity_normalization_scale": normalization_scale,
        "training_space_spatial_scaling": float(training.get("spatial_scaling")), "training_space_bbox_margin_mm": float(resolved.get("bbox", {}).get("margin_mm", 0.0)),
        "reconstruction_decoder_provenance_status": "fallback_config" if fallback_used else "optional_fields_present_or_not_required",
        "missing_optional_reconstruction_provenance": [key for key in ("source_mlp_checkpoint_metadata", "source_mlp_checkpoint_sha256", "scientific_checkpoint", "decoder_metadata") if key not in checkpoint],
    }
    return model, space, provenance


def export_anchor_parameters(
    output_path: str | Path,
    *,
    t1_ms: np.ndarray,
    t2_ms: np.ndarray,
    b1: np.ndarray,
    amplitude: np.ndarray,
    joint_dataset: Any,
    reconstruction_checkpoint: str | Path,
    geometry_checkpoint: str | Path,
    geometry_provenance: dict[str, Any],
    decoder_provenance: dict[str, Any] | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> Path:
    """Write an immutable NPZ plus JSON sidecar with the exact anchor contract."""
    values = {name: np.asarray(value, dtype=np.float32) for name, value in (("t1_ms", t1_ms), ("t2_ms", t2_ms), ("b1", b1), ("amplitude", amplitude))}
    count = int(torch.as_tensor(joint_dataset.xyz).shape[0])
    if any(value.shape != (count,) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"timing-only parameters must be finite anchor-aligned [N={count}]")
    anchor_hash = _joint_anchor_order_sha256(joint_dataset)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite timing-only parameter export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values, anchor_order_sha256=np.asarray(anchor_hash))
    provenance = {
        "schema": "ismrm_v2_2/timing_only_anchor_parameters/v3",
        "path": str(output), "sha256": _sha256_file(output),
        "reconstruction_checkpoint": str(Path(reconstruction_checkpoint).resolve()), "reconstruction_checkpoint_sha256": _sha256_file(Path(reconstruction_checkpoint)),
        "geometry_checkpoint": str(Path(geometry_checkpoint).resolve()), "geometry_checkpoint_sha256": _sha256_file(Path(geometry_checkpoint)),
        "post_hb1_geometry_sha256": geometry_provenance.get("post_hb1_geometry_sha256"), "selected_geometry_key": geometry_provenance.get("selected_key"),
        "trained_pose_key_not_used": "trained_axisangle_train", "anchor_order_sha256": anchor_hash, "anchor_count": count,
        "coordinate_contract": "fixed group_axisangle_post_stack_init_physical world RAS-mm anchor locations",
        "parameter_array_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in values.items()},
    }
    if decoder_provenance:
        provenance.update(decoder_provenance)
    if extra_provenance:
        provenance.update(extra_provenance)
    # Recompute the NPZ digest after all values have been written, and keep the
    # fixed identity fields authoritative even when callers provide extras.
    provenance["path"] = str(output)
    provenance["sha256"] = _sha256_file(output)
    provenance["anchor_order_sha256"] = anchor_hash
    provenance["anchor_count"] = count
    provenance["parameter_array_sha256"] = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in values.items()}
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def export_from_frozen_checkpoint(
    *,
    output_path: str | Path,
    joint_dataset: Any,
    scalar_dataset: Any,
    group_axisangle_physical: torch.Tensor,
    b6_model: str | Path,
    protocol_yaml: str | Path,
    reconstruction_checkpoint: str | Path,
    geometry_checkpoint: str | Path,
    geometry_provenance: dict[str, Any],
    signal_decoder_checkpoint: str | Path | None = None,
    provenance: dict[str, Any] | None = None,
    device: torch.device,
) -> Path:
    runtime_decoder = Path(signal_decoder_checkpoint).resolve() if signal_decoder_checkpoint is not None else None
    runtime_decoder_sha256 = None
    if runtime_decoder is not None:
        if not runtime_decoder.is_file():
            raise FileNotFoundError(f"runtime signal decoder checkpoint does not exist: {runtime_decoder}")
        runtime_decoder_sha256 = _sha256_file(runtime_decoder)
        if runtime_decoder_sha256 != APPROVED_DECODER_SHA256:
            raise ValueError(f"runtime signal decoder SHA256 is not the approved frozen decoder: {runtime_decoder}")
    # The audit exporter intentionally bypasses the production reconstruction
    # loader and never constructs the frozen signal simulator.
    model, _, checkpoint_provenance = load_frozen_quantitative_state_for_timing_audit(reconstruction_checkpoint, device=device, fallback_b6_model=b6_model)
    if "training_space_spatial_scaling" not in checkpoint_provenance:
        raise ValueError("frozen checkpoint lacks training-space reconstruction settings")
    current_space = TrainingSpace.from_dataset(scalar_dataset, float(checkpoint_provenance["training_space_spatial_scaling"]), group_axisangle_init_physical=group_axisangle_physical, bbox_margin_mm=float(checkpoint_provenance["training_space_bbox_margin_mm"]))
    model, space, checkpoint_provenance = load_frozen_quantitative_state_for_timing_audit(reconstruction_checkpoint, device=device, fallback_b6_model=b6_model, current_space=current_space)
    arrays = _dataset_arrays(joint_dataset, group_axisangle_physical)
    points = torch.as_tensor(arrays["physical_xyz"], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        fields = model(space.physical_ras_to_train(points))
    intensity_scale = torch.as_tensor(model.intensity_scale)
    if intensity_scale.numel() != 1 or not torch.isfinite(intensity_scale).all() or intensity_scale.item() <= 0:
        raise ValueError("frozen model intensity_scale must be one finite positive scalar for amplitude export")
    amplitude = fields["amplitude"] * intensity_scale
    decoder_provenance = {"runtime_decoder_path": str(runtime_decoder) if runtime_decoder else None, "runtime_decoder_sha256": runtime_decoder_sha256, "scientific_decoder_content_changed": False}
    extra_provenance = {
        "checkpoint_training_space_sha256": checkpoint_provenance.get("checkpoint_training_space_sha256"),
        "selected_geometry_key": geometry_provenance.get("selected_key"),
        "trained_pose_key_not_used": "trained_axisangle_train",
        "checkpoint_load": "audit-only quantitative checkpoint loaded",
        **dict(provenance or {}),
    }
    return export_anchor_parameters(output_path, t1_ms=fields["t1_ms"].detach().cpu().numpy(), t2_ms=fields["t2_ms"].detach().cpu().numpy(), b1=fields["b1"].detach().cpu().numpy(), amplitude=amplitude.detach().cpu().numpy(), joint_dataset=joint_dataset, reconstruction_checkpoint=reconstruction_checkpoint, geometry_checkpoint=geometry_checkpoint, geometry_provenance=geometry_provenance | checkpoint_provenance, decoder_provenance=decoder_provenance, extra_provenance=extra_provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True); parser.add_argument("--case", required=True); parser.add_argument("--prepared-root", required=True); parser.add_argument("--sparse-manifest"); parser.add_argument("--geometry-checkpoint", required=True); parser.add_argument("--b6-model", required=True); parser.add_argument("--protocol-yaml", required=True); parser.add_argument("--signal-decoder-checkpoint"); parser.add_argument("--reconstruction-checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    from ismrm_v2.proposed.q002.joint_dataset import JointAnchorDataset
    from ismrm_v2.proposed.quantitative_point_dataset import QuantPointDataset
    if args.sparse_manifest:
        inputs, _manifest = _manifest_active_inputs(args.sparse_manifest, subject=args.subject, case_id=args.case, prepared_root=args.prepared_root)
    else:
        from ismrm_v2.proposed.q002.contracts import cropped_observation_paths
        inputs = cropped_observation_paths(args.prepared_root, args.subject)
        _manifest = {}
    scalar = QuantPointDataset(inputs, device=device); joint = JointAnchorDataset(inputs, scalar)
    geometry, geometry_provenance = load_geometry_checkpoint(args.geometry_checkpoint, scalar.group_axisangle_init, scalar.group_resolution_xyz_mm)
    decoder_sha = _sha256_file(Path(args.signal_decoder_checkpoint).resolve()) if args.signal_decoder_checkpoint else None
    protocol_sha = _sha256_file(Path(args.protocol_yaml).resolve())
    git_sha = None
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    export_provenance = {"subject": args.subject, "requested_case": args.case, "resolved_case": _manifest.get("resolved_case", args.case) if args.sparse_manifest else args.case, "active_observation_records": _manifest.get("observation_records", {}) if args.sparse_manifest else {}, "sparse_working_manifest_sha256": _manifest.get("sha256") if args.sparse_manifest else None, "post_hb1_geometry_sha256": geometry_provenance.get("post_hb1_geometry_sha256"), "protocol_yaml": str(Path(args.protocol_yaml).resolve()), "protocol_sha256": protocol_sha, "approved_runtime_decoder_sha256": decoder_sha, "git_sha": git_sha}
    path = export_from_frozen_checkpoint(output_path=args.output, joint_dataset=joint, scalar_dataset=scalar, group_axisangle_physical=geometry, b6_model=args.b6_model, protocol_yaml=args.protocol_yaml, reconstruction_checkpoint=args.reconstruction_checkpoint, geometry_checkpoint=args.geometry_checkpoint, geometry_provenance=geometry_provenance, signal_decoder_checkpoint=args.signal_decoder_checkpoint, provenance=export_provenance, device=device)
    print(path)


if __name__ == "__main__":
    main()
