"""Exploratory myocardial residual dispersion and adjacent-ring contrast."""
import numpy as np
from scipy.ndimage import binary_dilation


def mechanism_statistics(reference, predictions, common, myocardium_core, myocardium_full):
    core = np.asarray(common, bool) & np.asarray(myocardium_core, bool)
    full = np.asarray(myocardium_full, bool)
    # Outer ring excludes the complete myocardium, not merely the eroded core.
    ring = binary_dilation(full, structure=np.ones((3, 3), bool), iterations=3) & ~full & np.asarray(common, bool)
    result = {"core_voxels": int(core.sum()), "peri_ring_voxels": int(ring.sum()), "methods": {}}
    for name, prediction in predictions.items():
        p = np.asarray(prediction, float); residual = p[core] - np.asarray(reference)[core]
        med = float(np.median(p[core])) if core.any() else None
        peri = float(np.median(p[ring])) if ring.any() else None
        result["methods"][name] = {"residual_mean": float(residual.mean()) if residual.size else None, "residual_sd": float(residual.std()) if residual.size else None, "residual_mad": float(np.median(np.abs(residual - np.median(residual)))) if residual.size else None, "myocardium_median_ms": med, "peri_ring_median_ms": peri, "contrast_abs_ms": abs(med - peri) if med is not None and peri is not None else None}
    return result
