# PIFNet

Multi-species protein function prediction. PIFNet predicts Gene Ontology (GO) terms for proteins by combining:

- **Sequence features:** per-residue embeddings from the protein language model ESM-2 (`esm2_t30_150M_UR50D`).
- **Network features:** protein embeddings learned from a multi-species protein–protein interaction (PPI) graph built from STRING, BioGRID and IntAct.
- **A GO-aware classifier:** each GO term attends over protein evidence to produce its score.
- **Optional homology blending:** model predictions can be blended with homology-based label transfer.

---

## Repository contents

| File | Purpose |
|---|---|
| `config.py` | Paths, species list, model and training settings |
| `data.py` | Loads dataset splits, labels, and information content (IC) |
| `esm_features.py` | Computes per-residue ESM-2 embeddings |
| `esm_store.py` | On-disk storage for ESM-2 embeddings |
| `ppi_graph.py` | Builds the multi-species PPI graph |
| `ppi_embed.py` | Learns node embeddings from the PPI graph |
| `models.py` | Model architecture |
| `losses.py` | Loss functions |
| `train.py` | Trains the model and writes predictions |
| `evaluate.py` | CAFA-style evaluation (Fmax, Smin, AUPR) |
| `homology.py` | Homology-based label transfer and blending |

---

## Dataset

### Source

Protein sequences, GO annotations and the train/validation/test splits are taken from the **MSNGO** benchmark:

> Beibei Wang *et al.* **MSNGO: multi-species protein function annotation based on 3D protein structure and network propagation.** arXiv:2503.23014 (2025).
> Code and data: https://github.com/blingbell/MSNGO


| Taxonomy ID | Organism |
|---|---|
| 3702 | *Arabidopsis thaliana* |
| 4577 | *Zea mays* |
| 284812 | *Schizosaccharomyces pombe* |
| 4932 | *Saccharomyces cerevisiae* |
| 6239 | *Caenorhabditis elegans* |
| 7227 | *Drosophila melanogaster* |
| 7955 | *Danio rerio* |
| 9606 | *Homo sapiens* |
| 10090 | *Mus musculus* |
| 10116 | *Rattus norvegicus* |
| 44689 | *Dictyostelium discoideum* |
| 511145 | *Escherichia coli* K-12 |
| 71421 | *Haemophilus influenzae* |

### Additional data

| Data | Source |
|---|---|
| STRING v12.0 (`protein.links` and `protein.aliases` per species) | https://string-db.org |
| BioGRID 5.0.255 | https://thebiogrid.org |
| IntAct | https://www.ebi.ac.uk/intact |
| Gene Ontology (`go-basic.obo`) | https://geneontology.org |
| ProtVec 3-gram vectors | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMFHTN |

## Usage

Set the data and output folders:

```bash
export PF_DATA_ROOT=/path/to/Data_msngo
export PF_WORK_ROOT=/path/to/Results
```

Run the pipeline for one ontology (`mf`, `bp` or `cc`):

```bash
BRANCH=mf

# 1. Build the PPI graph and learn network embeddings
python -m ppi_graph --branch $BRANCH --completion none
python -m ppi_embed --branch $BRANCH --completion none --method deepwalk

# 2. Compute ESM-2 embeddings
python -m esm_features --branch $BRANCH --split all --estimate-only   # check disk space first
python -m esm_features --branch $BRANCH --split all

# 3. Train and predict
python -m train --branch $BRANCH --fusion crossattn --seed 42 \
    --completion none --ppi-method deepwalk \
    --hidden-dim 512 --clf-hidden 512 --num-blocks 3 \
    --dilation-schedule aggressive --focal-alpha 0.75 --epochs 50

# 4. (Optional) Blend with homology-based predictions (requires DIAMOND)
python -m homology transfer --branch $BRANCH --threads 8
python -m homology blend    --branch $BRANCH --run <RUN_NAME>
```

Notes:
- Results are reported as the mean over seeds 42, 43 and 44.
- For BP, add `--no-hier-loss` to the training command to reduce GPU memory.
- `<RUN_NAME>` is the training run selected by validation Fmax.
- ESM-2 residue embeddings need a large amount of disk space (around 90 GB for the full dataset).
