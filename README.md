# FMO/PIEDA-supervised message-passing neural networks

This repository contains the code and reproduction metadata accompanying the manuscript “Extending Protein Representation Learning with Quantum Chemical Knowledge from the Fragment Molecular Orbital Method.”

The study investigates protein representation learning using fragment graphs with geometric pretraining (GEO) and additional supervision from FMO-derived pair interaction energy decomposition analysis (PIEDA). The primary GEO+PIEDA-C0 condition uses PIEDA only as a pretraining target and does not provide calculated PIEDA values to the encoder. The learned representations are evaluated by PIEDA readout and transfer to experimental activity prediction.

The exact source selection and checksums are recorded in SOURCE_PROVENANCE.json. Core authorship is recorded in AUTHORS.md, real-data execution checks in VALIDATION.md, and external input requirements in INPUTS.md. Raw FMO data, affinity values, graphs, and model weights are external inputs.

For data sources and exact placement, start with [INPUTS.md](INPUTS.md). SCOP2 has a cited public data deposit. The BindingDB-derived FMO collection has no verified public download in this release; its scientific citation is not presented as a download link. BDB-dependent workflows remain available for users who already have suitable inputs. Upstream FMO dataset reconstruction is outside this repository's scope.

The paper studies geometric pretraining (GEO) and additional PIEDA supervision without calculated PIEDA inputs (C0), with a surrounding-PIEDA input comparison (C1). ESM-FP, Unified-FP and Random-FP share the adopted structure/split assignments.

## Fixed inputs and settings

- [data/structures.csv](data/structures.csv): all 8,983 adopted structure keys in the original order, source, PDB ID, fixed-test and five-fold assignments, graph-content fingerprints and portable expected graph paths. BindingDB contributes 3,706 structures; SCOP2 contributes 5,277. The fixed test contains 370 and 528, respectively.
- [data/normalization.json](data/normalization.json): exact train-only means/stds and the ordered training-key hashes for final training and each fold.
- [data/schema.json](data/schema.json): node/edge schemas and target order. The graph fingerprints are tensor-content hashes from the original loader, not SHA-256 hashes of `.pt` files.
- [data/affinity_assignment.csv](data/affinity_assignment.csv): the adopted 3,135 labeled structure assignments (2,826 training-side, 309 test), endpoint types and recorded selection flags. Measured values are not distributed.
- [data/random_fp_codebook.json](data/random_fp_codebook.json): the actual fixed amino-acid bit codes, not a newly sampled replacement.
- [data/random_fp_manifests.json](data/random_fp_manifests.json): Random-FP-specific fingerprints and original manifest hashes; its splits and normalization were checked against the primary manifests.
- [configs/adopted_runs.json](configs/adopted_runs.json): seven adopted encoder configurations, seed 42, fixed epoch 300, source/config/manifest hashes and checkpoint hashes. URLs are left null where no approved artifact deposit exists.
- [data/input_assets.json](data/input_assets.json): external-input identities, availability and placement policy.
- [data/within_target_cohort.csv](data/within_target_cohort.csv): fixed memberships of the final 193 ligand groups across 64 targets, with original structure IDs and aggregation flags; no activity values or predictions.
- [data/test_target_annotations.csv](data/test_target_annotations.csv): target annotations for the original 309 test structures; distinct from the final ligand-group cohort.

## Materialize portable manifests

Python 3.10 or later; this metadata step uses only the standard library. Supply a directory containing `graphs/bdb/v6`, `graphs/bdb/v7`, `graphs/scop2/v6` and `graphs/scop2/v7`, using the relative names in `structures.csv`.

```bash
python prepare_inputs.py --graph-root /path/to/inputs --output /path/to/new-manifests --check-files
```

The command refuses an existing output directory. It generates the exact fixed assignments with user-supplied graph paths. Checking that a graph exists does not validate its tensor contents; the scientific loader must still check fingerprints and schema. No scheduler or institutional account is required for this metadata operation.

## Code and environment

- The reviewed scientific code is under [runtime/](runtime/). Graph construction and within-target evaluation also have top-level wrappers, described in [INPUTS.md](INPUTS.md) and below.
- The observed GPU environment is Python 3.10.12, PyTorch 2.5.1 with CUDA 12.1, and PyG 2.7.0. Use the versions listed in [requirements.txt](requirements.txt). This list is neither a historical lockfile nor a verified clean-environment installation.
- When running underlying workflow modules, set FMO_WORKDIR to the study working directory. The adopted label builder also uses FMO_BINDINGDB_CSV for the exact activity source. Some modules expect the original relative input names inside that directory; the top-level manifest generator does not provision all workflow inputs.

## Reproduction boundaries

Encoder training uses AdamW, learning rate 0.001, weight decay 0.0001, batch size 8, gradient clipping 5, and no learning-rate scheduler; condition-specific settings are in the adopted configs. Mask selection and initialization use the existing seed-derivation functions with base seed 42, rather than a replacement random-seeding scheme.

Frozen-encoder PIEDA head CV and encoder CV are different procedures. PIEDA head-only CV uses a fixed final encoder, five folds, up to 80 epochs, patience 10, Adam at 0.001 and batch size 4,096. Each readout/condition is refit for the median validation-selected epoch. pX heads use the adopted MLP/standardization, AdamW at 0.001 with weight decay 0.001, batch size 64, seed 42, and validation-selected epoch counts followed by refitting. These recipes still require the reviewed source and exact external inputs; Reproducing the reported numerical results additionally requires the external inputs described in [INPUTS.md](INPUTS.md).

## Within-target evaluation

Supply a CSV with `pdbid`, `target_px`, `GEO` and `GEO+PIEDA` predictions for the test structures. This command applies the recorded ligand grouping, verifies that pooled structures have the same observed pX, averages their predictions, centers observations/predictions separately within each target, and performs 10,000 paired target bootstrap draws with seed 42:

```bash
python evaluate_within_target.py --predictions /data/test_predictions.csv --cohort data/within_target_cohort.csv --out /data/new-within-target-results
```

Without `--cohort`, input rows must also include `target_group`; the evaluation then describes repeated structure-level target groups, rather than the final 193-ligand cohort. These are different analyses. [VALIDATION.md](VALIDATION.md) distinguishes execution-path checks from complete retraining and independent acquisition.

## License

The repository's original software is distributed under the [MIT License](LICENSE). External data, model weights, and installed dependencies retain their own terms.
