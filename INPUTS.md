# Existing external data: acquisition and placement

The repository supplies code and paper-specific IDs, splits, settings and normalization statistics. Obtain existing FMO data separately. Repeating upstream structural modeling, protonation or FMO calculations is outside the scope of this release.

## Sources

| Input | Source | Availability and role |
|---|---|---|
| SCOP2 FMO results and FMO input PDBs | [FMO-SCOP-29Jun2022, version 2](https://doi.org/10.6084/m9.figshare.25980112.v2), cited by [Takaya et al.](https://doi.org/10.1038/s41597-024-03999-2) | The paper's Data Records section identifies this figshare deposit and CC-BY terms. Obtain `mp2_631gdp_all.tsv.xz` and `pdbfile.tar`. Keep the downloaded files intact; check the deposit version and file identities. |
| BindingDB-derived FMO calculations | [Miyakawa et al., ChemRxiv version 1](https://doi.org/10.26434/chemrxiv.15009601/v1) | Scientific source, **not a verified download of the FMO dataset**. The received 26-page version describes the calculation procedure but supplies no dataset download identified in our review. The exact study inputs exist internally; public distribution of that collection has not been established. BDB-dependent paths require user-supplied existing data. |
| Experimental activities | [BindingDB official downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp) | Activity measurements are separate from FMO calculations. Downloading BindingDB does not provide FMO energies. Exact study input hashes are in `data/input_assets.json`; a newer monthly release need not yield the same labels. |
| General FMO repository | [FMODB](https://drugdesign.riken.jp/FMODB/), [download page](https://drugdesign.riken.jp/FMODB/download.php) | Existing calculation records can be downloaded. No verified mapping from the adopted BDB collection to FMODB records is supplied. A matching PDB ID alone does not establish matching modeled structures, fragments, calculation settings or energies; this is not presented as the adopted BDB download. |
| ESM features | `facebook/esm2_t12_35M_UR50D`, used by the supplied extraction code | ESM-FP uses 480 dimensions. The historical model revision is not pinned; adopted graph fingerprints identify the actual graph features. ESM weights are external dependencies. |

SCOP2 is the publicly sourced entry point. BindingDB-dependent paths are optional until the corresponding existing inputs are supplied. A missing BDB FMO download does not block a code-only release, but limits independent reproduction of the combined paper results. The repository does not promise to distribute unavailable upstream data or regenerate them.

## Directory layout

Choose an input root outside the checkout, for example `/data/fmo-inputs`:

```text
fmo-inputs/
  external/
    scop2/mp2_631gdp_all.tsv.xz
    scop2/pdbfile.tar
    bindingdb_fmo/                  # existing FMO inputs/logs/CPF, if available
    bindingdb_activity/             # downloaded activity files
  fragments/
    scop2/nodes/<key>_nodes.csv
    scop2/edges/<key>_edges.csv
    bdb/nodes/<pdb>_nodes.csv
    bdb/edges/<pdb>_edges.csv
  embeddings/<key>_esm.npy
  graphs/
    scop2/v6/<key>.pt
    scop2/v7/<key>.pt
    bdb/v6/<pdb>.pt
    bdb/v7/<pdb>.pt
  labels/affinity_labels_v2_adopted.csv
  checkpoints/geo/epoch_300.pt
  checkpoints/geo_pieda/epoch_300.pt
```

Downloaded TSV/PDB archives, fragment tables and `.pt` graphs are distinct formats. **Putting an archive under `external/` does not create graphs.** The supplied graph builders in `runtime/` consume fragment tables. Retain existing upstream fragment definitions and node/ESM row correspondence. This release does not undertake reconstruction of the upstream FMO dataset. Existing study-compatible graphs can be placed directly under `graphs/`; expected names and tensor fingerprints are in `data/structures.csv`.

A single prepared SCOP2 fragment-table pair can be processed independently of BindingDB, activity labels and encoder checkpoints:

```bash
python construct_graph.py --nodes /data/fmo-inputs/fragments/scop2/nodes/153l_a_nodes.csv --edges /data/fmo-inputs/fragments/scop2/edges/153l_a_edges.csv --key 153l_a --out /data/new-graphs/153l_a
```

This wrapper reuses the reviewed builders and v06prov QC: it keeps calculated non-neighbor PIEDA pairs without the dimer-ES approximation, removes QC outliers, and stores both edge directions. Add `--esm /path/to/153l_a_esm.npy` for the ESM-FP builder. The command does not perform full FPv2 adoption or certify the graph as a paper input. Downloading raw SCOP2 archives alone is not an end-to-end demonstration of training.

## Existing BDB calculation logs

When BDB FMO logs are already available, use them directly for validation.
This path does not require obtaining the dataset through the Miyakawa paper,
and does not repeat the upstream FMO calculation. Store the original log in
`external/bindingdb_fmo/` and retain its checksum. The supplied node parser
reads atomic coordinates, automatic fragmentation and fragment atom lists
from the ABINIT-MP log itself; a new PDB download is not required for that
step. Specify the actual target ligand residue name, because cofactors in
the log must not silently be treated as the target ligand.

For example, with an existing `1a27` log and target residue `EST`:

```bash
python runtime/create_fmo_fragment_nodes_from_log_v11.py --log-file /data/fmo-inputs/external/bindingdb_fmo/1a27_A_lastmodel_6n-flatmpi.log --pdbid 1a27 --ligand-resname EST --out-csv /data/fmo-inputs/fragments/bdb/nodes/1a27_nodes.csv --audit-json /data/fmo-inputs/fragments/bdb/nodes/1a27_audit.json
```

```bash
python runtime/create_fmo_fragment_edges_v2.py --log-file /data/fmo-inputs/external/bindingdb_fmo/1a27_A_lastmodel_6n-flatmpi.log --nodes /data/fmo-inputs/fragments/bdb/nodes/1a27_nodes.csv --out-csv /data/fmo-inputs/fragments/bdb/edges/1a27_edges.csv --exclude-dimer-es --require-pieda --exclude-neighbor-fragments
```

Pass these tables to `construct_graph.py`, using the ligand's canonical
isomeric SMILES via `--smiles`. Add `--paper-fpv2` to apply the adopted
ligand/water Morgan-FP assignment from the existing FPv2 implementation.
In particular, the older v6 builder alone leaves water FP zero; this is not
the adopted FPv2 input. This flag does not decide ligand identity or resolve
stereochemistry; the supplied ligand SMILES must already match the input.
ESM-FP additionally requires the existing
480-dimensional ESM array via `--esm`; verify node-row identities before
reusing that array. No measured activity values are needed to construct a
graph. pX fitting is a separate path that does require the activity labels.

## Fixed study inputs

Do not regenerate split membership from a seed against a different collection. The committed assignments cover 8,983 structures. With study-compatible graphs placed as above, materialize the exact six manifests:

```bash
python prepare_inputs.py --graph-root /data/fmo-inputs --output /data/new-manifests --check-files
```

Omit `--check-files` to prepare metadata with only SCOP2 present. This does not make combined training executable without BDB graphs. The scientific loader must still check tensor fingerprints and feature schema.

For BDB pX prediction, supply accepted labels separately; retain `data/affinity_assignment.csv` as the authoritative structure/endpoint/split assignment. Measured activities are not included. Seven encoder configurations and checkpoint hashes are in `configs/adopted_runs.json`; weights are optional external inputs, not included downloads.

FMO teacher QC and structure-only inference are distinct: GEO/GEO+PIEDA (C0) encoders do not receive calculated PIEDA inputs, while the reported performance uses already PIEDA-QC-filtered graphs. Wholly uncalculated structures do not inherit that QC claim.

See [VALIDATION.md](VALIDATION.md) for actual checks and their limits. Data availability, code licensing and numerical replication are reported separately.
