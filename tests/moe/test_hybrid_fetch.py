"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1


def test_frac_bs2_parse():
    from freetoken.moe.offload_kernels import _parse_frac_bs2

    assert _parse_frac_bs2(None) == 0.0
    assert _parse_frac_bs2("") == 0.0
    assert _parse_frac_bs2("0") == 0.0
    assert _parse_frac_bs2("0.25") == 0.25
    assert _parse_frac_bs2("1") == 1.0
    for bad in ("-0.1", "1.5", "nan", "abc"):
        with pytest.raises(ValueError):
            _parse_frac_bs2(bad)


def test_frac_bs2_only_replaces_a_benched_fraction(monkeypatch):
    from freetoken.moe import offload_kernels

    monkeypatch.setattr(offload_kernels, "_HYBRID_FRAC_BS2", 0.1)
    # explicit override at bs>=2 (takes precedence over the margin); bs=1 keeps the benched value
    assert offload_kernels._fetch_fraction_for(8, 0.3) == 0.1
    assert offload_kernels._fetch_fraction_for(1, 0.3) == 0.3
    # no benched fraction (explicit --moe-hybrid-max-fetch / no profile): the fixed cap stays
    assert offload_kernels._fetch_fraction_for(8, 0.0) == 0.0


def test_parse_margin():
    from freetoken.moe.offload_kernels import _parse_margin

    assert _parse_margin(None) == 1.0   # default: no margin (opt in per hardware)
    assert _parse_margin("") == 1.0
    assert _parse_margin("0.5") == 0.5  # enable: halve the benched fraction at bs>=2
    assert _parse_margin("0.25") == 0.25
    for bad in ("0", "-0.1", "1.5", "abc"):
        with pytest.raises(ValueError):
            _parse_margin(bad)


def test_bs2_margin_default_off_and_opt_in(monkeypatch):
    from freetoken.moe import offload_kernels

    monkeypatch.setattr(offload_kernels, "_HYBRID_FRAC_BS2", 0.0)  # no explicit override
    # default (margin 1.0): behaviour unchanged -- bs>=2 keeps the benched fraction
    monkeypatch.setattr(offload_kernels, "_HYBRID_BS2_MARGIN", 1.0)
    assert offload_kernels._fetch_fraction_for(8, 0.30) == 0.30
    # opted in (margin 0.5): halved at bs>=2, bs=1 and fixed caps untouched
    monkeypatch.setattr(offload_kernels, "_HYBRID_BS2_MARGIN", 0.5)
    assert offload_kernels._fetch_fraction_for(8, 0.30) == 0.15
    assert offload_kernels._fetch_fraction_for(2, 0.138) == 0.069
    assert offload_kernels._fetch_fraction_for(1, 0.30) == 0.30
    assert offload_kernels._fetch_fraction_for(8, 0.0) == 0.0


def test_hybrid_fixed_cap_unchanged_at_bs2_with_frac_bs2(monkeypatch):
    from freetoken.moe import offload_kernels

    monkeypatch.setattr(offload_kernels, "_HYBRID_FRAC_BS2", 0.1)
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cpu"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=4,
    )
    ids = torch.arange(16, dtype=torch.int32).view(2, 8)
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 16
    assert int(cache.num_indices.item()) == 4


def test_lfu_decays_small_counts(monkeypatch):
    # A once-hot expert must age out: with integer freq, freq - (freq >> 3) never decays a
    # count below 8, so an expert routed by 7 tokens once would outrank one used last step.
    from freetoken.moe import offload_kernels

    monkeypatch.setattr(offload_kernels, "_HYBRID_EVICT_BY_LFU", True)
    monkeypatch.setattr(offload_kernels, "_HYBRID_FRAC_BS2", 0.0)
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=4, device=torch.device("cpu"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=4,
    )
    a, b, c, d = 0, 1, 2, 3

    def step(layer, ids):
        cache.ensure_experts_hybrid(layer, torch.tensor(ids, dtype=torch.int32).view(-1, 1))

    step(0, [a] * 7)  # a: 7 routes, once
    for _ in range(40):
        step(0, [c, c])
    step(0, [b, c])   # b: 1 route, last step
    step(1, [a, a])   # fill the last slot from another layer
    assert int((cache.id_of_slot >= 0).sum()) == 4
    step(0, [d, c])   # d needs a slot: evict the stale a, keep the recent b
    assert int(cache.slot_for_id[0, a]) == -1
    assert int(cache.slot_for_id[0, b]) >= 0
    assert int(cache.slot_for_id[0, d]) >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fetch,evict", [("multiplicity", "lfu"), ("recency", "lru")])
def test_hybrid_batched_gpu_matches_cpu_reference(monkeypatch, fetch, evict):
    from freetoken.moe import offload_kernels

    monkeypatch.setattr(offload_kernels, "_HYBRID_FETCH_BY_MULT", fetch == "multiplicity")
    monkeypatch.setattr(offload_kernels, "_HYBRID_FETCH_BY_RECENCY", fetch == "recency")
    monkeypatch.setattr(offload_kernels, "_HYBRID_EVICT_BY_LFU", evict == "lfu")
    monkeypatch.setattr(offload_kernels, "_HYBRID_FRAC_BS2", 0.0)
    monkeypatch.setattr(offload_kernels, "_HYBRID_BS2_MARGIN", 1.0)  # test parity at the raw fraction
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    # skewed (without-replacement) routing so multiplicities differ and hot experts recur
    popularity = torch.arange(num_experts, 0, -1, dtype=torch.float32) ** 2
    for step in range(96):
        bs = (1, 2, 8)[step % 3]  # mixed batch sizes exercise the bs=1 <-> bs>=2 switch
        layer = step % 2
        ids = torch.multinomial(popularity.expand(bs, -1), top_k).to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()
        gpu.ensure_experts_hybrid(layer, g)
        ref.ensure_experts_hybrid(layer, c)
        assert int(gpu.num_missing_full.item()) == int(ref.num_missing_full.item())
        assert int(gpu.num_indices.item()) == int(ref.num_indices.item())
        assert torch.equal(g.cpu(), c)
        for name in ("slot_for_id", "id_of_slot", "usage", "expert_recency", "expert_freq"):
            assert torch.equal(getattr(gpu, name).cpu(), getattr(ref, name).cpu()), name
