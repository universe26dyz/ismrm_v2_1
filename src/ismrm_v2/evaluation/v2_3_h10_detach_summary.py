"""Generate subject, cohort and paired-error summaries from completed evaluations."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

METHODS = ("torch_mse", "torch_hgnll10", "torch_hgnll10_detach")
METRICS = ("rmse", "mae", "nrmse", "bias", "ncc", "cc", "psnr")
COMPARISONS = {"h10_vs_mse": (METHODS[0], METHODS[1]), "detach_vs_h10": (METHODS[1], METHODS[2]), "detach_vs_mse": (METHODS[0], METHODS[2])}


def summarize(input_root, output):
    paths = sorted(Path(input_root).rglob("manifest.json"))
    roots = []
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("schema") == "v2_3_h10_detach_native/v1": roots.append((path.parent, data))
    if not roots: raise ValueError("no completed H10 detach evaluations found")
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    rows = []; seen = set()
    for root, manifest in roots:
        subject = manifest["subject"]
        if subject in seen: raise ValueError(f"duplicate subject {subject}")
        seen.add(subject)
        report = json.loads((root / "aggregate_metrics.json").read_text())
        for parameter, partitions in report["partitions"].items():
            for partition, domains in partitions.items():
                for domain, result in domains.items():
                    for method in METHODS:
                        m = result["methods"][method]
                        rows.append({"subject": subject, "parameter": parameter, "partition": partition, "domain": domain, "method": method, "support_voxels": result["support_voxels"], **{k: m.get(k) if m else None for k in METRICS}})
    accum = defaultdict(list); lookup = {}
    for row in rows:
        key = tuple(row[k] for k in ("subject", "parameter", "partition", "domain"))
        lookup[(key, row["method"])] = row
        for metric in METRICS:
            value = row[metric]
            if value is not None: accum[(row["parameter"], row["partition"], row["domain"], row["method"], metric)].append(value)
    means = [{"parameter": p, "partition": part, "domain": d, "method": m, "metric": metric, "mean": sum(v)/len(v), "n_subjects": len(v)} for (p, part, d, m, metric), v in accum.items()]
    deltas = []
    for key in sorted({key for key, _ in lookup}):
        for label, (baseline, candidate) in COMPARISONS.items():
            for metric in ("rmse", "mae", "nrmse"):
                a, b = lookup[(key, baseline)][metric], lookup[(key, candidate)][metric]
                deltas.append(dict(zip(("subject", "parameter", "partition", "domain"), key)) | {"comparison": label, "metric": metric, "baseline": a, "candidate": b, "improvement_percent": 100 * (a-b)/a if a is not None and b is not None and a != 0 else None})
    output.mkdir(parents=True)
    for name, values in (("per_subject_metrics.csv", rows), ("cohort_mean_metrics.csv", means), ("paired_method_deltas.csv", deltas)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]) if values else []); writer.writeheader(); writer.writerows(values)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input-root", required=True); parser.add_argument("--output", required=True)
    print(summarize(**vars(parser.parse_args())))

if __name__ == "__main__": main()
