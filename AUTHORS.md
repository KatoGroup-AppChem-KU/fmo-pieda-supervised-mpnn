# Code authorship

For the following four core modules, the original implementer confirmed the
attribution through the project owner on 25 November 2026:

> Original implementation: Koichiro Kato
>
> Subsequent development and modifications: Hirokazu Okamoto

- `audit_fmo_pieda_v06_provisional_exclusion.py`
- `build_fmo_fragment_pyg_graph_v7_morganunified.py`
- `geopair_cfg_20261006.py`
- `pretrain_masked_edge_multigraph_v17r_rawdist.py`

This resolves their previously unconfirmed authorship. Historical folder names
and original provenance records describe the earlier state and are retained.
The adopted pretraining snapshot differs from the older local mirror; its
SHA-256 is `d766501fd72dcc4c4a14c1a872e158710921f49025f6faf09880788b21633b1b`.

The new `evaluate_within_target.py` entry point is derived from Hirokazu
Okamoto's `within_target_affinity_20261105.py`. It retains separate centering
of observations and predictions, exact-ID-set grouping, and the original
10,000-replicate cluster bootstrap with seed 42.

`construct_graph.py` is a thin wrapper by Hirokazu Okamoto around the reviewed
builders and QC implementation; their original authorship is preserved. Its
optional ligand/water FP assignment is derived from Okamoto's
`rebuild_bdb_fp_v2_20261016.py`, rather than a new fingerprint recipe.

The project's original software is licensed under [MIT](LICENSE). Upstream data and installed dependencies retain
their own terms and are not relicensed by this repository.
