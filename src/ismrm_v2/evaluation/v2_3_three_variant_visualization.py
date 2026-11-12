"""Three-variant native-plane figures using the legacy SCMR map LUTs."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from ismrm_v2.evaluation.visualization import _crop_bounds, colorbars
from ismrm_v2.evaluation.scmr.quality_control import DISPLAY_RANGES_MS, RESIDUAL_RANGES_MS


def render_plane(reference, predictions, support, cardiac_roi, parameter, output, *, title=""):
    """Render one matched native plane; crop is shared by every map and residual."""
    import matplotlib.pyplot as plt
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import METHODS
    reference = np.asarray(reference)
    support = np.asarray(support, bool)
    cardiac_roi = np.asarray(cardiac_roi, bool)
    if cardiac_roi.shape != reference.shape or support.shape != reference.shape:
        raise ValueError("cardiac crop and support must match native plane")
    bounds = _crop_bounds(cardiac_roi, margin=3)
    r0, r1, c0, c1 = bounds
    sl = np.s_[r0:r1, c0:c1]
    bars = colorbars(); key = parameter.upper(); limit = RESIDUAL_RANGES_MS[key][1]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    fig.suptitle(title)
    values = [reference] + [predictions[name] for name in METHODS]
    for ax, label, value in zip(axes[0], ("Native 2D", "Torch MSE", "HGNLL-10", "HGNLL-1"), values):
        image = ax.imshow(np.ma.masked_where(~support[sl], np.asarray(value)[sl]), origin="lower", interpolation="nearest", cmap=bars.mapping[key], vmin=DISPLAY_RANGES_MS[key][0], vmax=DISPLAY_RANGES_MS[key][1]); ax.set_title(label); ax.axis("off")
    fig.colorbar(image, ax=list(axes[0]), label=f"{key} (ms)", shrink=.8)
    axes[1, 0].axis("off")
    for ax, name in zip(axes[1, 1:], METHODS):
        delta = np.asarray(predictions[name]) - reference
        image_delta = ax.imshow(np.ma.masked_where(~support[sl], delta[sl]), origin="lower", interpolation="nearest", cmap="coolwarm", vmin=-limit, vmax=limit); ax.set_title(name + " − native"); ax.axis("off")
    fig.colorbar(image_delta, ax=list(axes[1, 1:]), label=f"Δ{key} (ms)", shrink=.8)
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white"); plt.close(fig)
    return {"path": str(path), "crop_bounds_rc": list(bounds), "map_range_ms": list(DISPLAY_RANGES_MS[key]), "residual_range_ms": [-limit, limit], "residual_colormap": "coolwarm (signed adaptation of legacy absolute residual range)", "colorbars": bars.metadata}


def render_sax_reformats(cache, images, roi_by_index, output):
    """Reformat the SAX stack on shared physical centre planes, as in Final5."""
    from ismrm_v2.evaluation.final_cyj_reporting import _sax_stack_affine_ras, _sax_cut_plane, _dense_physical_plane
    from ismrm_v2.evaluation.evaluation import _world_sample
    from ismrm_v2.evaluation.v2_3_three_variant_native_metrics import METHODS
    if len(cache["group_idx"]) < 2:
        return []  # A single SAX slice cannot span a longitudinal plane.
    if not roi_by_index:
        raise ValueError("SAX reformat requires at least one mapped cardiac ROI")
    shape = tuple(cache["source_support"].shape[1:])
    roi_volume = np.zeros((shape[0], shape[1], len(cache["group_idx"])), bool)
    for i, mask in roi_by_index.items(): roi_volume[:, :, i] = mask
    rows, cols = np.where(roi_volume.any(axis=2))
    if not rows.size: raise ValueError("empty SAX cardiac ROI")
    affine = _sax_stack_affine_ras(cache)
    records = []
    for cut, index in (("center_row", int(np.median(rows))), ("center_column", int(np.median(cols)))):
        source_plane = _sax_cut_plane((*shape, len(cache["group_idx"])), affine, cut=cut, index=index)
        plane = _dense_physical_plane(source_plane)
        cardiac, cardiac_valid = _world_sample(roi_volume.astype(float), affine, plane["world"], order=0)
        crop = (cardiac > .5) & cardiac_valid
        if not crop.any(): continue
        # Fill between annotated SAX levels for crop only; do not invent metric support.
        rr, cc = np.where(crop); crop[rr.min():rr.max()+1, cc.min():cc.max()+1] = True
        for parameter in ("t1", "t2"):
            native = np.moveaxis(np.asarray(cache[parameter + "_ms"], float), 0, -1)
            native_mask = np.moveaxis(np.asarray(cache["source_support"], bool), 0, -1)
            ref, good = _world_sample(native, affine, plane["world"], order=0)
            valid_mask, vg = _world_sample(native_mask.astype(float), affine, plane["world"], order=0)
            common = good & vg & (valid_mask > .5) & np.isfinite(ref)
            preds = {}
            for name in METHODS:
                image = images[name][parameter]
                values, valid = _world_sample(image.get_fdata(), image.affine, plane["world"], order=1)
                preds[name] = values; common &= valid & np.isfinite(values)
            if not common.any(): continue
            target = Path(output) / f"sax_{cut}_{parameter}.png"
            info = render_plane(ref, preds, common, crop, parameter, target, title=f"SAX {cut} reformat — {parameter.upper()}")
            info.update({"cut": cut, "index": index, "parameter": parameter, "spacing_mm": plane["actual_spacing_mm"], "geometry": "shared physical SAX centre plane; native nearest, volumes linear"})
            records.append(info)
    return records
