"""
Central config for the grokking data pipeline.
All hyperparameters live here so train.py / sweep.py / analysis.py stay in sync.
Recipe follows Nanda et al. (arXiv:2301.05217). See docs/BLUEPRINT.md §6a.
"""
from pathlib import Path

# --- Task -------------------------------------------------------------------
P = 113                      # modulus:  compute a + b (mod P)
VOCAB_SIZE = P + 1           # numbers 0..112 plus the '=' token
EQUALS_TOKEN = P             # id of '='

# --- Model (1-layer transformer, no LayerNorm) ------------------------------
N_LAYERS = 1
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
ACT = "relu"
USE_LAYERNORM = False

# --- Mainline training ------------------------------------------------------
TRAIN_FRAC = 0.30
WEIGHT_DECAY = 1.0
LR = 1e-3
BETAS = (0.9, 0.98)
OPTIMIZER = "AdamW"
FULL_BATCH = True            # 1 optimization step == 1 epoch
STEPS = 40_000
SEED = 0

# --- Grok definition --------------------------------------------------------
GROK_ACC = 0.99             # val accuracy threshold
GROK_SUSTAIN = 10           # consecutive evals above threshold

# --- Eval / checkpoint schedule ---------------------------------------------
N_EVAL_STEPS = 400          # ~log-spaced eval points, 0..STEPS
N_CKPT_STEPS = 30           # ~log-spaced full-weight checkpoints (for spectrum-over-time)
SNAPSHOT_LABELS = [         # the 5 embedding/prediction snapshot points
    "init", "memorized", "forming", "grokked", "final",
]

# --- Sweep (Act II) ---------------------------------------------------------
SWEEP_FRACTIONS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]
SWEEP_WEIGHT_DECAYS = [0.0, 0.1, 0.3, 1.0, 3.0]
SWEEP_BUDGET_STEPS = 30_000
SWEEP_N_EVAL = 150

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT.parent / "web" / "public" / "data"
RUNS_DIR = OUT_DIR / "runs"
REPORT = ROOT / "report.md"

SCHEMA_VERSION = "1.0"
