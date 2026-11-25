# Real-data validation and limits

On 25 November 2026 the selected 60-module portable candidate was placed in
an isolated directory. Only candidate scientific modules were imported;
external installed libraries and existing data/checkpoints were permitted.
Imported candidate file locations and SHA-256 hashes were checked. The
original private scientific source directories were not import fallbacks.
Execution used Slurm partition `aurum-g01`, Python 3.10.12 and the existing
CUDA environment. This was not a fresh dependency installation.

| Path | Actual check |
|---|---|
| Graph construction | Existing real `1a27` fragment-node/edge CSVs and ESM features; adopted three pair-selection rules and v06prov QC; v6 and v7 builders each produced 289 nodes and 3,786 directed edges. Both outputs had identical edge topology/features and finite expected feature dimensions. This does not certify every FPv2 node feature or repeat upstream FMO calculations. |
| BDB log-to-update path | Existing `1a27` and `1a9u` ABINIT-MP logs were parsed by candidate node/edge parsers, then passed through QC/builders and the adopted ligand/water FP assignment. Both v6/v7 node-feature tensors matched adopted inputs exactly. Fragment IDs and topology matched exactly; maximum edge-feature difference was 9.54e-7 from normalization roundoff. These log-derived graphs passed the unchanged matched optimizer/checkpoint smoke with deterministic PyTorch operations. |
| SCOP2 without BindingDB | Public `construct_graph.py` on existing real `153l_a` SCOP2 fragment tables, with neither BDB input nor checkpoint/ESM weights. Unified-FP output: 183 nodes, 546 node features and 2,574 directed edges after the adopted selection/QC. Input was an existing fragment-table pair; this did not demonstrate converting the downloaded figshare archives to tables. |
| GEO / GEO+PIEDA (C0) pretraining | Existing adopted smoke routine on eight real training graphs: two optimizer steps per condition; matched initialization/masks, no FMO/q input, bidirectional geometry masking, GEO invariance to PIEDA teachers, gradient routing, unchanged frozen parameters and checkpoint continuation parity. Initialization matched the adopted run. |
| Frozen representation extraction | Adopted epoch-300 encoders loaded strictly. Twenty-four real BDB/SCOP2 graphs, eight each from train/validation/test, passed content-fingerprint/schema checks. Node/pair readout inputs were extracted. Encoder parameters stayed frozen and unchanged. |
| PIEDA readout | Original fitting functions; node and pair probes for both conditions, two epochs each. Train/validation/test retained 15,892 / 15,272 / 14,546 directed readout edges. Finite five-component predictions and metrics were saved. |
| pX prediction | Twenty-four actual accepted structures, including original train/validation/test membership. Node, pair and concatenated representations; original MLP fitting function, two epochs per condition/readout. Finite predictions and separate split metrics were saved. |
| Within-target evaluation | Public standalone evaluation reproduced the stored structure-level 309-test analysis after centering its 207 repeatedly represented structures across 71 exact ID sets. A separate check uses the paper's final 193-ligand/64-target memberships. These populations must not be interchanged. |

The short training checks establish that the processing paths execute. They
do not reproduce 300 epochs, the full five-fold head search or all paper
metrics. Frozen extraction used already trained study checkpoints; the two
newly smoke-trained encoders were not substituted for them. Within-target
numerical comparison used stored study predictions, not the short-run pX
heads. Exact results, selected IDs and imported-module hashes are retained
in the associated validation records.

The first two-log GPU smoke stopped at the existing bitwise GEO-invariance
assertion under ordinary CUDA operations. Repeating the unchanged smoke with
`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
passed. The final check also applied the existing FPv2 ligand/water assignment;
the older bare v6 constructor by itself omits water FP. Neither model/loss/mask
code nor the assertion tolerance was loosened.

The final cohort-membership wrapper reproduced the stored 193-ligand,
64-target concatenated-representation analysis: GEO r = 0.6093050930,
GEO+PIEDA (C0) r = 0.7110561795, difference 0.1017510865 with paired
95% CI [0.0225932671, 0.1899133887]. Its comparison is recorded separately.
The observations here are ligand groups; they are not 193 independent target
groups. Public accessibility of BDB FMO data and pretrained weights is not
established by an internal successful run.
