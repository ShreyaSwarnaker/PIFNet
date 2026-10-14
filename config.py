"""
Central configuration. Edit DATA_ROOT and the species list for a new dataset;
everything else has sane defaults.
"""
import os
from dataclasses import dataclass, field
from typing import List

_CPU = os.cpu_count() or 8

# Hardcoded to your actual layout, so no env vars or sourcing are needed.
# PF_DATA_ROOT / PF_WORK_ROOT still override these if you ever set them
# (e.g. to point at a second dataset without editing this file).
DATA_ROOT = os.environ.get(
    "PF_DATA_ROOT", "./Data_msngo")
WORK_ROOT = os.environ.get(
    "PF_WORK_ROOT", "./Results")

# NCBI taxonomy IDs for the STRING species you downloaded.
# Add/remove freely -- the code derives filenames from these.
#
# MSNGO's 13 training species (TAXIDS in blingbell/MSNGO utils.py), plus two
# alternative ids for organisms whose STRING taxid differs between releases:
# 4932 vs 559292 (S. cerevisiae) and 511145 vs 83333 (E. coli K-12). Listing
# both is harmless -- ppi_graph.py skips files that are absent and only keeps
# edges whose endpoints map into your protein set.
SPECIES_MSNGO = ["3702", "4577", "284812", "559292", "6239", "7227", "7955",
                 "9606", "10090", "10116", "44689", "83333", "71421",
                 "4932", "511145"]

# The 7 species used for the earlier CAFA-5 run.
SPECIES_CAFA5 = ["9606", "10090", "4932", "7227", "10116", "511145", "3702"]

# The 23 species represented in the CAFA3 target set.
# These are the NCBI taxonomy IDs listed in the official CAFA3
# target tracking sheet.
SPECIES_CAFA3 = [
    "7227",
    "208964",
    "237561",
    "9606",
    "243232",
    "273057",
    "160488",
    "171101",
    "223283",
    "224308",
    "243273",
    "99287",
    "511145",
    "85962",
    "10090",
    "10116",
    "284812",
    "3702",
    "44689",
    "4932",
    "7955",
    "8355",
]

SPECIES = SPECIES_MSNGO

# MSNGO built its networks from STRING v11.0. Use "11.0" for a like-for-like
# comparison with their published numbers; "12.0" is current and has more
# edges, which is a data advantage a reviewer would notice.
STRING_VERSION = "12.0"


@dataclass
class Paths:
    data_root: str = DATA_ROOT
    work_root: str = WORK_ROOT
    ontology: str = os.path.join(DATA_ROOT, "..", "go-basic.obo")
    protvec: str = os.path.join(DATA_ROOT, "protVec_100d_3grams.csv")
    biogrid: str = os.path.join(DATA_ROOT, "BIOGRID-ALL-5.0.255.tab3.txt")
    intact: str = os.path.join(DATA_ROOT, "intact.txt")

    def split_csv(self, branch: str, split: str) -> str:
        return os.path.join(self.data_root, f"{branch}_{split}.csv")

    def ic_csv(self, branch: str) -> str:
        return os.path.join(self.data_root, f"{branch}_ic.csv")

    string_version: str = STRING_VERSION

    def string_links(self, taxid: str) -> str:
        return os.path.join(
            self.data_root,
            f"{taxid}.protein.links.v{self.string_version}.txt")

    def string_aliases(self, taxid: str) -> str:
        return os.path.join(
            self.data_root,
            f"{taxid}.protein.aliases.v{self.string_version}.txt")

    def work(self, *parts) -> str:
        p = os.path.join(self.work_root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p


@dataclass
class GraphConfig:
    string_threshold: int = 400        # STRING combined_score cutoff (0-1000)
    knn_k: int = 15                    # sequence neighbours per protein
    alpha: float = 0.5                 # preferential-attachment exponent
    # "triadic": for edge (a,b) and c ~ a, add (c,b)  -- uses graph structure
    # "knn":     for edge (a,b) and c ~ a, add (a,c)  -- pure sequence kNN (original code)
    completion_mode: str = "triadic"
    max_new_edges_per_node: int = 10    # partners sampled per source node
    # Global budget: keep only the highest-weight proposed edges, up to this
    # fraction of the observed edge count. Prevents completion from swamping
    # measured interactions (the original run added ~26%).
    new_edge_budget: float = 0.30


@dataclass
class EmbedConfig:
    dim: int = 128
    walk_number: int = 10
    walk_length: int = 80
    window: int = 5
    epochs: int = 5                    # was 1 in the notebook -- too few
    workers: int = 8
    project_top_k: int = 30            # top-k refs for inductive projection
    project_temperature: float = 0.1   # softmax temperature over similarities


@dataclass
class ModelConfig:
    esm_model: str = "esm2_t30_150M_UR50D"
    esm_layer: int = 30
    esm_dim: int = 640
    max_len: int = 1024
    hidden_dim: int = 256
    ppi_dim: int = 128
    dropout: float = 0.3
    # Stacked multi-scale blocks. 1 block = 17-residue receptive field (the
    # original, far too small for 100-300aa domains); 3 blocks = 337 residues.
    num_blocks: int = 3
    dilation_schedule: str = "aggressive"   # moderate | aggressive
    # gate | concat | crossattn | seq_only | ppi_only
    fusion: str = "gate"


@dataclass
class TrainConfig:
    # 128 x 1 accumulation == the old 64 x 2, i.e. identical effective batch
    # of 128, so optimisation is unchanged but there are half as many kernel
    # launches. Raise batch_size further if the GPU is still idle -- but if you
    # change the EFFECTIVE batch (batch_size x accumulation_steps) you are
    # changing the experiment and should re-tune lr.
    batch_size: int = 128
    eval_batch_size: int = 256
    accumulation_steps: int = 1
    prefetch_factor: int = 2       # lower RAM footprint than 4; each worker
                                    # buffers this many batches ahead
    persistent_workers: bool = True  # keep worker processes alive between epochs
                                      # instead of respawning them each epoch --
                                      # meaningful over 30-50 epochs
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 50      # was 30; best epoch was 28/30, i.e. still improving
    patience: int = 10
    # SAFE-BY-DEFAULT, not auto-scaled to total machine cores. Auto-scaling
    # to os.cpu_count() was wrong: it doesn't know how many OTHER jobs are
    # running at the same time (run_staged.sh / run_ablations.py launch
    # several concurrently, each independently thinking it can claim ~12
    # workers), and on a shared box with other users' jobs, that oversubscribes
    # CPU/RAM badly -- workers get OOM-killed and DataLoader respawns them in
    # a loop, which looks like "stuck" training, not faster training.
    # Override explicitly (train.py --num-workers) only when you know you are
    # the sole job on the box.
    num_workers: int = 2
    seed: int = 42
    hierarchical_loss: bool = True
    # alpha now genuinely balances classes (see losses.FocalLoss). With ~99.5%
    # negative entries, positives need alpha > 0.5 to be up-weighted.
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    focal_balanced: bool = True
    # Threads for evaluate_cafa's threshold sweep (called every epoch). Same
    # oversubscription problem as num_workers if left auto-scaled: capped low
    # and NOT tied to os.cpu_count() for the same reason.
    eval_threads: int = 4
    # Threshold grid for the per-epoch validation eval used to pick the best
    # checkpoint. Final reported metrics always use the CAFA-standard 101.
    eval_thresholds_train: int = 26
    amp: bool = True                   # mixed precision -- big speed win


@dataclass
class Config:
    branch: str = "mf"                 # mf | bp | cc
    paths: Paths = field(default_factory=Paths)
    graph: GraphConfig = field(default_factory=GraphConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    species: List[str] = field(default_factory=lambda: list(SPECIES))

    ROOT_GO = {"mf": "GO:0003674", "bp": "GO:0008150", "cc": "GO:0005575"}
    NAMESPACE = {
        "mf": "molecular_function",
        "bp": "biological_process",
        "cc": "cellular_component",
    }

    @property
    def root_go(self) -> str:
        return self.ROOT_GO[self.branch]

    @property
    def namespace(self) -> str:
        return self.NAMESPACE[self.branch]


def get_config(**overrides) -> Config:
    cfg = Config()

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Select branch-specific work directory unless explicitly overridden.
    if "PF_WORK_ROOT" not in os.environ:
        cfg.paths.work_root = os.path.join(
            ".",
            f"Results_{cfg.branch}"
        )

    # Fail loudly instead of silently resolving paths under the wrong root.
    # This is what happened when PF_DATA_ROOT/PF_WORK_ROOT were left unset in
    # a fresh shell: everything quietly wrote to ./work instead of Results/,
    # and every "missing file" error looked like a real bug.
    if not os.path.isdir(cfg.paths.data_root):
        raise FileNotFoundError(
            f"PF_DATA_ROOT resolves to '{cfg.paths.data_root}', which does not "
            "exist. Set PF_DATA_ROOT to your data folder (e.g. export PF_DATA_ROOT=/path/to/Data_msngo). "
            "(Env vars do not carry over between terminal sessions or between "
            "a script's subprocess and your interactive shell.)"
        )
#    if os.path.basename(os.path.normpath(cfg.paths.data_root)) == "Camp_5":
#        cfg.species = list(SPECIES_CAFA5)
        
    data_name = os.path.basename(os.path.normpath(cfg.paths.data_root))

    if data_name == "Camp_5":
        cfg.species = list(SPECIES_CAFA5)
    
    elif data_name == "cafa_3":
        cfg.species = list(SPECIES_CAFA3)
        
    return cfg
    