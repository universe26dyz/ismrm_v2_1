"""Export frozen-geometry, case-specific T1/T2/B1 parameters for timing-only audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cross_view_reliability_audit import _dataset_arrays, _joint_anchor_order_sha256, _sha256_file, load_geometry_checkpoint
from .same_world_reliability_audit import _manifest_active_inputs

APPROVED_DECODER_SHA256 = "338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b"


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
        "schema": "ismrm_v2_2/timing_only_anchor_parameters/v2",
        "path": str(output), "sha256": _sha256_file(output),
        "reconstruction_checkpoint": str(Path(reconstruction_checkpoint).resolve()), "reconstruction_checkpoint_sha256": _sha256_file(Path(reconstruction_checkpoint)),
        "geometry_checkpoint": str(Path(geometry_checkpoint).resolve()), "geometry_checkpoint_sha256": _sha256_file(Path(geometry_checkpoint)),
        "post_hb1_geometry_sha256": geometry_provenance.get("post_hb1_geometry_sha256"), "selected_geometry_key": geometry_provenance.get("selected_key"),
        "trained_pose_key_not_used": "trained_axisangle_train", "anchor_order_sha256": anchor_hash, "anchor_count": count,
        "coordinate_contract": "fixed group_axisangle_post_stack_init_physical world RAS-mm anchor locations",
    }
    if decoder_provenance:
        provenance.update(decoder_provenance)
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
    signal_decoder_checkpoint: str | Path,
    device: torch.device,
) -> Path:
    from ismrm_v2.proposed.mlp_trainer import build_mlp_training_model, load_mlp_reconstruction_checkpoint
    from ismrm_v2.proposed.q001_config import load_b6_resolved_config
    runtime_decoder = Path(signal_decoder_checkpoint).resolve()
    if not runtime_decoder.is_file():
        raise FileNotFoundError(f"runtime signal decoder checkpoint does not exist: {runtime_decoder}")
    runtime_decoder_sha256 = _sha256_file(runtime_decoder)
    if runtime_decoder_sha256 != APPROVED_DECODER_SHA256:
        raise ValueError(f"runtime signal decoder SHA256 is not the approved frozen decoder: {runtime_decoder}")
    config = load_b6_resolved_config(b6_model)
    historical_decoder_path = config.get("decoder", {}).get("checkpoint") if isinstance(config.get("decoder"), dict) else None
    config = copy.deepcopy(config)
    config["decoder"] = dict(config.get("decoder", {}))
    config["decoder"]["checkpoint"] = str(runtime_decoder)
    model, space, _protocol, _metadata = build_mlp_training_model(scalar_dataset, config, protocol_yaml, device, initial_group_axisangle_physical=group_axisangle_physical)
    checkpoint = load_mlp_reconstruction_checkpoint(reconstruction_checkpoint, model, device)
    arrays = _dataset_arrays(joint_dataset, group_axisangle_physical)
    points = torch.as_tensor(arrays["physical_xyz"], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        fields = model.inr(space.physical_ras_to_train(points))
    intensity_scale = torch.as_tensor(model.intensity_scale)
    if intensity_scale.numel() != 1 or not torch.isfinite(intensity_scale).all() or intensity_scale.item() <= 0:
        raise ValueError("frozen model intensity_scale must be one finite positive scalar for amplitude export")
    amplitude = fields["amplitude"] * intensity_scale
    return export_anchor_parameters(output_path, t1_ms=fields["t1_ms"].detach().cpu().numpy(), t2_ms=fields["t2_ms"].detach().cpu().numpy(), b1=fields["b1"].detach().cpu().numpy(), amplitude=amplitude.detach().cpu().numpy(), joint_dataset=joint_dataset, reconstruction_checkpoint=reconstruction_checkpoint, geometry_checkpoint=geometry_checkpoint, geometry_provenance=geometry_provenance | {"checkpoint_load": "frozen reconstruction checkpoint loaded", "trained_pose_not_selected": True, "checkpoint_metadata_keys": sorted(checkpoint)}, decoder_provenance={"historical_decoder_path_from_b6": historical_decoder_path, "runtime_decoder_path": str(runtime_decoder), "runtime_decoder_sha256": runtime_decoder_sha256, "scientific_decoder_content_changed": False})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True); parser.add_argument("--case", required=True); parser.add_argument("--prepared-root", required=True); parser.add_argument("--sparse-manifest"); parser.add_argument("--geometry-checkpoint", required=True); parser.add_argument("--b6-model", required=True); parser.add_argument("--protocol-yaml", required=True); parser.add_argument("--signal-decoder-checkpoint", required=True); parser.add_argument("--reconstruction-checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--device", default="cpu")
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
    scalar = QuantPointDataset(inputs, device=device); joint = JointAnchorDataset(inputs, scalar)
    geometry, geometry_provenance = load_geometry_checkpoint(args.geometry_checkpoint, scalar.group_axisangle_init, scalar.group_resolution_xyz_mm)
    path = export_from_frozen_checkpoint(output_path=args.output, joint_dataset=joint, scalar_dataset=scalar, group_axisangle_physical=geometry, b6_model=args.b6_model, protocol_yaml=args.protocol_yaml, reconstruction_checkpoint=args.reconstruction_checkpoint, geometry_checkpoint=args.geometry_checkpoint, geometry_provenance=geometry_provenance, signal_decoder_checkpoint=args.signal_decoder_checkpoint, device=device)
    print(path)


if __name__ == "__main__":
    main()
