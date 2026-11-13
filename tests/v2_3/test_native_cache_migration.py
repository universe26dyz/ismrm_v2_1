import copy
import json

import numpy as np
import pytest

from ismrm_v2.evaluation.final_cyj_reporting import _load_all_verified_native_cache_groups
from ismrm_v2.fit_first.contracts import sha256
from ismrm_v2.fit_first.fit_first import (
    NATIVE_FIT_METHOD, SCMR_V5_NATIVE_FIT_METHOD, SCMR_V5_DICTIONARY_CONTRACT,
    _dictionary_contract,
)


@pytest.fixture
def native_cache(tmp_path):
    root = tmp_path / "native"; cache = root / "sax"; cache.mkdir(parents=True)
    current = tmp_path / "current.npz"; historical = tmp_path / "historical.npz"
    masks = np.ones((20, 3, 3), bool); masks[3, 1, 1] = False
    np.savez(current, group_idx=np.repeat([0, 1], 10), weight_idx=np.tile(np.arange(10), 2), masks=masks, timing9_ms=np.zeros((20, 9)), tr_ms=np.full(20, 2.61), vps=np.full(20, 87))
    historical.write_bytes(current.read_bytes())
    np.savez(cache / "native_maps.npz", t1_ms=np.ones((2, 3, 3)), t2_ms=np.ones((2, 3, 3)), b1=np.ones((2, 3, 3)), group_idx=np.array([0, 1]))
    np.savez(cache / "geometry.npz", affine_lps_rc=np.repeat(np.eye(4)[None], 2, axis=0), pixel_spacing_rc_mm=np.ones((2, 2)), slice_thickness_mm=np.ones(2))
    record = {"method": NATIVE_FIT_METHOD, "dictionary_contract": _dictionary_contract(), "native_maps_sha256": sha256(cache / "native_maps.npz"), "geometry_sha256": sha256(cache / "geometry.npz"), "source_observations": {"path": str(historical), "sha256": sha256(current)}}
    sparse = {"status": "ELIGIBLE", "stacks": {"sax": {"source_observation": {"path": str(current), "sha256": sha256(current)}}}}
    sparse_path = tmp_path / "sparse.json"
    def save():
        (cache / "manifest.json").write_text(json.dumps(record))
        sparse_path.write_text(json.dumps(sparse))
    save()
    return {"root": root, "cache": cache, "current": current, "historical": historical, "record": record, "sparse": sparse, "sparse_path": sparse_path, "save": save}


def load(fixture):
    fixture["save"]()
    return _load_all_verified_native_cache_groups(fixture["root"], fixture["sparse_path"], "sax")


@pytest.mark.parametrize("scmr", [False, True])
def test_supported_contracts_and_identical_sources(native_cache, scmr):
    f = native_cache
    if scmr:
        f["record"]["method"] = SCMR_V5_NATIVE_FIT_METHOD
        f["record"]["dictionary_contract"] = copy.deepcopy(SCMR_V5_DICTIONARY_CONTRACT)
    result = load(f)
    assert result["source_support"].shape == (2, 3, 3)
    assert not result["source_support"][0, 1, 1] and result["source_support"][1].all()
    provenance = result["provenance"]
    assert provenance["source_observations"] == {"path": str(f["current"].resolve()), "sha256": sha256(f["current"])}
    resolution = provenance["source_observation_resolution"]
    assert resolution == {"selected_from": "sparse_manifest", "cache_declared_path": str(f["historical"]), "sparse_declared_path": str(f["current"]), "declared_sha256_equal": True}


def test_migrated_missing_historical_path(native_cache):
    native_cache["historical"].unlink()
    assert load(native_cache)["provenance"]["source_observation_resolution"]["selected_from"] == "sparse_manifest"


def test_missing_sparse_path_uses_cache(native_cache):
    native_cache["current"].unlink()
    result = load(native_cache)["provenance"]
    assert result["source_observation_resolution"]["selected_from"] == "cache_manifest"
    assert result["source_observations"]["path"] == str(native_cache["historical"].resolve())


def test_source_identity_mismatch(native_cache):
    native_cache["sparse"]["stacks"]["sax"]["source_observation"]["sha256"] = "different"
    with pytest.raises(ValueError, match="identities differ"): load(native_cache)


@pytest.mark.parametrize("path_key", ["current", "historical"])
def test_existing_source_hash_mismatch_fails_even_for_unselected_path(native_cache, path_key):
    native_cache[path_key].write_bytes(b"changed")
    with pytest.raises(ValueError, match="source observation SHA256 mismatch"): load(native_cache)


def test_both_sources_missing(native_cache):
    native_cache["current"].unlink(); native_cache["historical"].unlink()
    with pytest.raises(ValueError, match="source observations are unavailable"): load(native_cache)


def test_unknown_contract_fails(native_cache):
    native_cache["record"]["dictionary_contract"]["matching"] = "other"
    with pytest.raises(ValueError, match="frozen dictionary contract"): load(native_cache)


@pytest.mark.parametrize("file", ["native_maps.npz", "geometry.npz"])
def test_cache_hash_mismatch_fails(native_cache, file):
    (native_cache["cache"] / file).write_bytes(b"modified")
    with pytest.raises(ValueError, match="native cache SHA256 mismatch"): load(native_cache)
