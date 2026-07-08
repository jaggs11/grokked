# GROKKED — Build Blueprint

**Working title:** *Grokked — watching a neural network stop memorizing and start understanding.*
**Purpose of this document:** the complete, buildable architecture for the interactive explainer. Drop this file into the repo as `docs/BLUEPRINT.md`. Both Claude Code prompts (§6) instruct the agent to read specific sections of this file as source of truth.

**How to use it:** read §0–§1 to internalize the concept, §5 to internalize the story (this is your interview script), then run prompt 6(a) in a Python/GPU environment and prompt 6(b) in the Next.js repo.

---

## 0. Executive decisions — where I pushed back

Your brief was strong. Here is what I changed and why, up front, so you can veto anything before the build:

1. **Replaced the abstract "Memorization vs. Generalization side-by-side" with a concrete Prediction Map.** A conceptual lookup-table-vs-circle graphic is the weakest kind of explainer content — illustrative, not evidential. Instead: a 113×113 grid of every possible `(a, b)` pair, colored by whether the model currently gets it right, with training pairs outlined. At the memorization checkpoint you see green confetti (the train set) drowning in red; at the grok step the whole grid floods green. Same lesson, but it's *real model output*, it's beautiful, and it's ~2 KB of data per checkpoint. This becomes an Act I centerpiece, not a sidebar.

2. **The Hero Curve gets an "x-ray mode" showing the paper's actual progress measures.** The single most impressive thing you can show a frontier-lab person is not the accuracy cliff — it's the *evidence that the circuit forms gradually while accuracy is flat*: key-frequency energy rising, excluded loss diverging, weight norm falling during cleanup. These are literally what Paper 2 is titled after ("Progress Measures"). They ride as toggleable ghost traces on the hero chart. This is the beat that proves "sudden outside, gradual inside" quantitatively instead of asserting it.

3. **Upgraded the Clock to a five-dial array with a constructive-interference readout.** A single clock is a nice metaphor but it undersells (and slightly misrepresents) the mechanism: the network uses *several* frequencies, and the answer is read out as a sum of cosines that spikes at the correct `c` and cancels everywhere else — a small DFT. So: intro with one clock for intuition, then reveal five dials ticking at different rates, then the interference plot `Σₖ αₖ cos(wₖ(a+b−c))` peaking at the answer. The interference plot is the deepest single visual in the piece and it's cheap to render client-side from a tiny fitted-coefficients file.

4. **Constrained the Ablation Playground to the honestly precomputable space.** Free-form toggling of 56 frequencies implies 2⁵⁶ combinations you can't precompute, and faking interpolated numbers violates your honesty constraint. The honest design that still *feels* free: (a) toggle any *single* frequency (56 real measurements), (b) toggle any *subset of the five key frequencies* with noise on or off (2⁵ × 2 = 64 real measurements), (c) three hero presets. Every number on screen is a real forward-pass result. The UI communicates the constraint as a feature ("every configuration here was actually run").

5. **Phase-Transition Map ships as a coarse grid with honest "did-not-grok" cells, with a 1-D "cross" fallback.** A dense fraction×weight-decay grid is your single biggest compute line-item. Plan A: 7 fractions × 5 weight decays, one seed, 30k-step cap with early stopping — cells that never grok are rendered as a distinct "DNF within budget" state, which is itself the finding. Plan B (if GPU time is short): two 1-D sweeps through the mainline point, rendered as one connected cross-shaped view. Both are specced in the data contract.

6. **Shared training-step state is scoped, not global, and has an explicit story/manual mode machine.** Syncing *everything* to one scrubbed step creates jank and a confusing mental model. Instead: the step is global state, but only step-aware views subscribe (Hero, Prediction Map, Spectrum). Scroll drives the step in "story mode"; the moment the reader grabs the scrubber, the view enters "manual mode" and scroll stops fighting them until the next section boundary, where a quiet "resync" occurs. The Ablation Playground is pinned to the final checkpoint by design — making ablations step-dependent multiplies precompute 30× for near-zero narrative gain.

7. **Added a short Coda + Methods footer, and an interview fact sheet (Appendix A).** The footer is where intellectual honesty lives (run config, seed, "all numbers precomputed from checkpoints, nothing is simulated live"), and it doubles as a credibility signal to exactly the audience you're targeting.

Everything else in your brief survives intact.

---

## 1. Product blueprint

### 1.1 Concept and emotional arc

One page, one story, three acts, one instrument.

The through-line is a **mystery structure**: Act I makes the reader *feel* an anomaly (a model that looks permanently stuck, then snaps), Act II proves the anomaly is *lawful* (it responds predictably to data and regularization — so there must be a mechanism), Act III *opens the machine* and finds actual mathematics inside — then replays Act I's curve with x-ray vision so the reader sees that the "sudden" jump was the last visible step of a long hidden construction. The final emotional beat is a reframe: **the click you saw was demolition, not construction** — weight decay tearing down the memorization scaffolding around a circuit that had quietly finished forming.

The reader should leave able to say, in one breath: *"It memorized first because that's easy; a Fourier-rotation algorithm formed slowly underneath because weight decay charges rent and the algorithm is cheaper; the visible jump is the cleanup when the scaffolding comes down."*

### 1.2 View-by-view verdicts

- **Hero Curve — KEEP, extended.** Train/val accuracy on log-x steps, draggable scrubber, phase bands, plus x-ray toggles (key-frequency energy fraction, excluded loss, restricted loss, total weight norm — normalized to a shared 0–1 right axis with per-trace tooltips giving true values). The scrubber position is the global `stepIdx`.
- **Prediction Map — NEW (replaces Memorization-vs-Generalization).** Canvas grid of all `(a,b)` pairs, correct/incorrect at the current step, train cells outlined. Synced to `stepIdx`.
- **Phase-Transition Map — KEEP, de-risked** per §0.5. Clicking a cell loads that run's slim curve as an overlay in a compare panel (mainline shown as a ghost), rather than hijacking the hero — the hero stays the story's spine.
- **Fourier Embedding View — KEEP.** Two linked panels: raw embedding heatmap (looks like static) and the Fourier-norm spectrum (five spikes). A checkpoint slider (synced to `stepIdx`, snapped to available checkpoints) animates noise → spikes. Raw heatmap ships at 5 canonical checkpoints only (payload discipline); the spectrum animates across ~30 checkpoints.
- **Clock / Rotation View — KEEP, upgraded** per §0.3: one intro dial → five-dial array → interference readout. The dial angles are the *actual learned* embedding angles per frequency (shipped in `clock.json`), not idealized ones — quietly labeled as such.
- **Ablation Playground — KEEP, constrained** per §0.4.
- **Neuron Gallery — KEEP as progressive disclosure.** A collapsed "for the curious" strip of ~12 MLP-neuron activation heatmaps over the (a,b) grid, each tagged with its dominant frequency — the plaid patterns are gorgeous and cost ~150 KB. First candidate to cut (§7).
- **Memorization vs. Generalization conceptual graphic — CUT** (absorbed by Prediction Map).

### 1.3 Design language — "the instrument on the bench"

The trap for this project is the default AI-generated look (near-black page, one neon accent, glassy cards). The subject hands us something much better: this is a story about **measuring a hidden periodic signal** — the natural world of chart recorders, spectrum analyzers, and lab notebooks.

**The signature element:** the page itself is light — cool drafting-paper (`#ECEEE9`) with a faint blue-grey plot grid and ink-dark prose — but every visualization lives inside **one persistent dark viewport, "the instrument"**: a deep blue-black scope (`#0C1220`) with a thin bezel, an always-visible monospace **step odometer** (top-right, ticking as you scrub or scroll), and a tiny provenance badge (bottom-right, `◦ from checkpoint data` → tooltip with run id). The duality *is* the thesis: the paper page is the outside view, the scope is the window into the black box. As the reader scrolls, text steps flow past on paper while the instrument stays pinned and its display changes.

**Palette (semantic, not decorative):**

| Token | Hex | Meaning |
|---|---|---|
| `--paper` | `#ECEEE9` | page background |
| `--ink` | `#1A1F2A` | prose, axes on paper |
| `--scope` | `#0C1220` | instrument background |
| `--train` | `#F0B24A` | training-set traces (warm amber) |
| `--val` | `#5FE3C0` | validation traces (phosphor green-cyan) |
| `--signal` | `#B48CFF` | key frequencies / the circuit (violet) |
| `--noise` | `#5A6478` | non-key frequencies, memorization residue (slate) |

Train vs. val are additionally distinguished by dash pattern (never color alone). Phase bands: three translucent tints of `--noise`, `--signal`, `--val` respectively — memorization is noise-colored, formation is signal-colored, cleanup is val-colored, so the *legend teaches the story*.

**Type:** display & narrative prose in **Spectral** (a screen-first serif whose name is a gift for this subject); data, axes, and the odometer in **IBM Plex Mono**; small UI labels in **Archivo** (compact, instrument-panel register). Type scale: 1.25 ratio, prose measure ~65ch.

**Motion principles:** every animation encodes a data change or a narrative beat; nothing is ambient. Durations 250–600 ms, one custom ease (`cubic-bezier(0.22, 1, 0.36, 1)`); springs only on the scrubber handle. Scroll-linked motion uses `transform`/`opacity` only. `prefers-reduced-motion`: all autoplay and scroll-driven animation replaced by stepped crossfades with visible next/prev controls — the story remains fully traversable.

**One aesthetic risk, justified:** the hero cold-open (§5, S0) lets the plateau *take real time* — the trace draws slowly while the odometer spins through thousands of steps, deliberately testing the reader's patience for ~4 seconds before the cliff. Boredom-as-design, because the boredom is the phenomenon. (Reduced-motion and a "skip" affordance included.)

---

## 2. Information architecture & folder structure

One repo, two worlds: `pipeline/` (Python, run offline) writes JSON into `web/public/data/`, which the app treats as read-only truth.

```
grokked/
├── docs/
│   └── BLUEPRINT.md                  # this file
├── pipeline/                         # offline; never deployed
│   ├── README.md
│   ├── requirements.txt
│   ├── config.py                     # p, model dims, hyperparams, seeds, paths
│   ├── model.py                      # 1-layer transformer (Nanda architecture)
│   ├── train.py                      # mainline run + checkpointing
│   ├── sweep.py                      # fraction × weight-decay grid (+ --quick cross mode)
│   ├── fourier.py                    # Fourier basis over Z_p, projections, key-freq detection
│   ├── analysis.py                   # progress measures, ablations, clock data, neurons,
│   │                                 #   prediction maps, phase-boundary detection
│   ├── export.py                     # writes all JSON per §4; quantization + RLE helpers
│   ├── validate_contract.py          # jsonschema validation of every emitted file
│   └── checkpoints/                  # .pt files (gitignored)
└── web/
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx                  # the single page; composes acts
    │   ├── globals.css               # tokens from §1.3
    │   └── opengraph-image.tsx       # hero curve as OG card
    ├── components/
    │   ├── stage/                    # ScrollStage, Instrument, StepText, SectionRail
    │   ├── hero/                     # HeroCurve, PhaseBands, Scrubber, XRayToggles, Odometer
    │   ├── grids/                    # PredictionMap, NeuronGallery (canvas)
    │   ├── sweep/                    # PhaseMap, SweepCompare
    │   ├── fourier/                  # EmbeddingHeatmap, FourierSpectrum, CheckpointSlider
    │   ├── clock/                    # ClockIntro, DialArray, DialFace, InterferenceReadout
    │   ├── ablation/                 # AblationBoard, FreqToggleGrid, PresetRow, MetricReadout
    │   └── ui/                       # DataBadge, Kbd, Toggle, MethodsFooter
    ├── lib/
    │   ├── data/                     # loaders.ts, schemas.ts (zod mirrors of §4), rle.ts, quant.ts
    │   ├── state/store.ts            # zustand store (§3.2)
    │   ├── d3/scales.ts              # shared log-step scale, color scales
    │   └── motion/                   # variants.ts, useReducedMotion.ts, useScrollStep.ts
    ├── public/data/                  # §4 files; runs/ subfolder for slim sweep curves
    └── (next.config, tsconfig, tailwind config…)
```

Deployment: Vercel, fully static (`export` where possible). All JSON served from `/public/data` with long-cache headers; heavy files lazy-loaded per act with prefetch-on-approach.

---

## 3. Component tree & shared state

### 3.1 Component inventory

**Stage**

- `ScrollStage` — owns the scrollytelling pattern per act: a sticky `Instrument` beside/behind a column of `StepText` blocks; wires an IntersectionObserver-based `useScrollStep` hook that reports the active step. Props: `{ sections: SectionDef[], children }`.
- `Instrument` — the persistent dark viewport: bezel, `Odometer`, `DataBadge` slot, and a content slot that crossfades between visualizations as sections change. Props: `{ view: ReactNode, runLabel: string }`.
- `StepText` — one narrative step; declares its choreography. Props: `{ id, stepKeyframe?: number, onEnter?: (store) => void, children }`. `stepKeyframe` is a target `stepIdx` the store animates to in story mode.
- `SectionRail` — left-edge progress rail with act/section ticks; keyboard `1–9` jumps (also powers "interview mode", §7).

**Hero**

- `HeroCurve` — D3-in-React line chart. Log-x (steps, clamped ≥1), left y = accuracy, faint right y = normalized x-ray traces. Renders `PhaseBands`, train/val paths, optional x-ray paths, and the `Scrubber`. Props: `{ run: RunMetrics, phases?: PhaseBoundaries, xray: Set<XRayKey>, ghostRun?: SlimRun }`. Subscribes to `stepIdx` transiently (no React re-render per frame; direct DOM/canvas update).
- `Scrubber` — a styled `<input type="range">` over the log axis (index-based, so keyboard arrows step through eval points). Emits `setStep(idx, 'user')`.
- `XRayToggles` — legend-as-controls; each x-ray trace's legend chip is its toggle (`aria-pressed`).
- `Odometer` — monospace step counter with rolling digits; reads `stepIdx` transiently.

**Grids**

- `PredictionMap` — canvas, one pixel-block per `(a,b)` cell; correct → `--val`, incorrect → dim red-slate, train cells get a 1px `--train` outline (togglable). Snaps `stepIdx` to nearest available prediction checkpoint and crossfades pre-rendered offscreen buffers. Props: `{ maps: PredictionMaps }`.
- `NeuronGallery` — collapsed-by-default grid of small canvas multiples; hover shows neuron id + dominant frequency; clicking a frequency chip highlights that neuron's clan. Props: `{ neurons: NeuronGrid[] }`.

**Sweep**

- `PhaseMap` — SVG/canvas heatmap; x = train fraction, y = weight decay (log-ish ordinal), color = `log10(grok_step)`, hatched cells = `not_grokked`, `◦` marker on the mainline cell. Click → `selectRun(cell)`. Props: `{ sweep: Sweep, selected?: CellRef }`.
- `SweepCompare` — mini accuracy chart: selected cell's slim curve solid, mainline as ghost; states for `not_grokked` (flat line annotated "no grok within 30k steps"). Props: `{ run?: SlimRun, ghost: SlimRun }`.

**Fourier**

- `EmbeddingHeatmap` — canvas, `p × d_model` int8-dequantized matrix at the nearest snapshot checkpoint, diverging colormap. Props: `{ snapshots: EmbeddingSnapshots }`.
- `FourierSpectrum` — bar spectrum of per-frequency embedding norm; key frequencies tinted `--signal` once detected; animates across ~30 checkpoints following `stepIdx`. Props: `{ spectrum: SpectrumOverTime, highlight?: number }`. Hovering a bar sets `hoveredFreq` (cross-highlights DialArray & AblationBoard if mounted).
- `CheckpointSlider` — thin snapping slider under Fourier views mirroring the global scrubber but constrained to checkpoint steps.

**Clock**

- `ClockIntro` — single SVG dial, numbers 0…112 as ticks; given `(a,b)`, animates the hand: rotate to `a`, then advance by `b`, land on `(a+b) mod p`. Props: `{ a, b, p }`.
- `DialArray` — five `DialFace`s, one per key frequency `k`; each shows token positions at the *learned* angles for that frequency; the same `(a,b)` composes at `k×` speed on each dial. Props: `{ clock: ClockData, a, b }`.
- `InterferenceReadout` — line/area plot of `logit(c) ≈ Σₖ αₖ cos(wₖ(a+b−c))` over all `c`, spiking at the true sum; per-dial contribution toggles show the peak sharpening as dials stack. Displays the pipeline-computed fit R² in a `DataBadge`. Computed client-side from `clock.json` coefficients (pure trig — honest, since coefficients are measured).
- Shared `(a,b)` picker: two small number steppers + "random pair" button; state local to the clock section.

**Ablation**

- `AblationBoard` — orchestrates board state; resolves the current toggle configuration to a *precomputed* result row (never interpolates); if a configuration isn't in the table (can't happen through the UI, but guard anyway) shows an honest "not measured" state. Props: `{ ablations: Ablations }`.
- `FreqToggleGrid` — 56 chips; key-frequency chips (violet) are individually toggleable; noise chips toggle as a single "noise ×51" group (this is what makes the space precomputable, and the grouping itself teaches sparsity). Single-noise-chip toggles allowed too (56 single-ablation rows exist).
- `PresetRow` — three buttons: *Keep only the 5 key frequencies* / *Delete only the 5 key frequencies* / *Delete all 51 noise frequencies*.
- `MetricReadout` — big odometer-style val loss + val accuracy, with baseline deltas; loss bar annotated with `ln(113) ≈ 4.727` "coin-flip ceiling" line.

**UI**

- `DataBadge` — the honesty mechanism: `◦ from checkpoint data` / `◦ fitted approximation (R² shown)` / `◦ our reproduction` variants; tooltip carries run id + config from `manifest.json`.
- `MethodsFooter` — run card (model dims, optimizer, seed, hardware, commit), paper citations, link to pipeline code, limitations paragraph.
- `Toggle`, `Kbd` — primitives.

### 3.2 Shared state model (zustand)

```ts
type StepSource = 'scroll' | 'user' | 'sync';
type Mode = 'story' | 'manual';

interface ExplainerStore {
  // global training-step cursor over the mainline eval grid
  stepIdx: number;                    // index into mainline.steps
  mode: Mode;
  setStep: (idx: number, source: StepSource) => void;
  // rule: source 'user'  -> mode = 'manual'
  //       source 'scroll'-> applied only if mode === 'story'
  //       source 'sync'  -> applied always; mode = 'story' (section-boundary resync)

  section: string;                    // active StepText id (from useScrollStep)
  setSection: (id: string) => void;   // on section change: if mode==='manual', resync()

  activeRun:                          // what SweepCompare shows
    | { kind: 'mainline' }
    | { kind: 'sweep'; frac: number; wd: number; seed: number };
  selectRun: (r: ExplainerStore['activeRun']) => void;

  hoveredFreq: number | null;         // cross-highlight between Spectrum / Dials / Ablation
  reducedMotion: boolean;
}
```

Performance contract: canvas/D3 components subscribe to `stepIdx` via `store.subscribe` (transient) and mutate the DOM/canvas inside `requestAnimationFrame` — React re-renders happen only on section changes, run selection, or toggle changes. Animated step transitions (story mode keyframes) tween `stepIdx` through the store so *all* subscribed views move together.

---

## 4. The data contract

All files live in `web/public/data/`. Conventions: arrays are index-aligned with their `steps`/`checkpoint_steps` companions; losses are natural-log cross-entropy; accuracies in `[0,1]`; every file carries `schema_version` and `run_id`. The pipeline's `validate_contract.py` and the frontend's zod schemas must both encode exactly what follows.

**Encodings.** `int8-quant`: `{ "scale": number, "data_b64": string }`, value = `int8 × scale`, row-major. `rle-bitmask`: run-length counts alternating starting with the count of 0s, row-major over the `p×p` grid, e.g. `[3, 5, 2, …]`.

### 4.1 `manifest.json`

```jsonc
{
  "schema_version": "1.0",
  "generated_at": "2026-07-12T…Z",
  "p": 113,
  "vocab_size": 114,                       // 0..112 plus '='
  "chance_accuracy": 0.00885,
  "uniform_loss": 4.7274,                  // ln(113)
  "mainline": {
    "run_id": "mainline_p113_f0.30_wd1.0_s0",
    "train_frac": 0.30, "n_train": 3831, "n_val": 8938,
    "weight_decay": 1.0, "lr": 0.001, "betas": [0.9, 0.98],
    "optimizer": "AdamW", "batch": "full", "epochs": 40000, "seed": 0,
    "model": { "n_layers": 1, "d_model": 128, "n_heads": 4,
               "d_head": 32, "d_mlp": 512, "act": "relu", "layernorm": false }
  },
  "key_freqs": [14, 35, 41, 42, 52],       // DETECTED from this run, not hardcoded
  "key_freq_detection": "top-k Fourier norm of W_E at final checkpoint; k chosen at the elbow",
  "phase_boundaries": { "memorization_end": 1400,   // DETECTED (see §6a); values here illustrative
                        "formation_end": 9400, "cleanup_end": 14000 },
  "grok_step": 11000,                       // first step with val_acc ≥ .99 sustained 10 evals
  "grok_step_def": "val_acc>=0.99 for 10 consecutive evals",
  "logit_fit_r2": 0.0,                      // filled by pipeline
  "files": { "mainline": "mainline.json", "sweep": "sweep.json",
             "spectrum": "spectrum_over_time.json", "embeddings": "embedding_snapshots.json",
             "clock": "clock.json", "ablations": "ablations.json",
             "predictions": "prediction_maps.json", "neurons": "neurons.json" },
  "provenance": { "code_commit": "…", "torch": "…", "gpu": "…", "wall_clock_min": 0 }
}
```

### 4.2 `mainline.json` — hero curves + progress measures

```jsonc
{
  "schema_version": "1.0",
  "run_id": "mainline_p113_f0.30_wd1.0_s0",
  "steps": [0, 1, 2, 3, 5, 7, 10, …, 40000],   // ~400 log-spaced eval steps (dense early)
  "train_acc": [ … ], "val_acc": [ … ],
  "train_loss": [ … ], "val_loss": [ … ],
  "progress": {
    "key_freq_frac": [ … ],     // fraction of W_E Fourier norm in key freqs (final-ckpt key set)
    "gini": [ … ],              // concentration of Fourier norms across freqs
    "restricted_loss": [ … ],   // val loss keeping ONLY key-freq logit components (Nanda def.)
    "excluded_loss": [ … ],     // TRAIN loss with key-freq logit components removed (Nanda def.)
    "w_norm": [ … ]             // sqrt of total sum-of-squares of all parameters
  }
}
```

All arrays share `steps.length`. Frontend normalizes each x-ray trace to `[0,1]` for the shared right axis and shows true values in tooltips.

### 4.3 `sweep.json` + `runs/*.json` — phase map

```jsonc
{
  "schema_version": "1.0",
  "mode": "grid",                                  // or "cross" (fallback)
  "fractions": [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70],
  "weight_decays": [0.0, 0.1, 0.3, 1.0, 3.0],
  "budget_steps": 30000,
  "grok_step_def": "val_acc>=0.99 for 10 consecutive evals",
  "cells": [
    { "frac": 0.30, "wd": 1.0, "seed": 0, "status": "grokked",
      "grok_step": 11000, "final_val_acc": 0.999,
      "curve": "runs/f0.30_wd1.0_s0.json" },
    { "frac": 0.15, "wd": 0.0, "seed": 0, "status": "not_grokked",
      "grok_step": null, "final_val_acc": 0.011,
      "curve": "runs/f0.15_wd0.0_s0.json" }
    // …
  ]
}
```

Slim per-run file `runs/f{frac}_wd{wd}_s{seed}.json`: `{ "steps": [~150 log-spaced], "train_acc": [...], "val_acc": [...] }` — nothing else, so each is a few KB and lazy-loads on click.

### 4.4 `spectrum_over_time.json`

```jsonc
{
  "schema_version": "1.0",
  "freqs": [0, 1, 2, …, 56],                 // 0 = DC; 56 = (p−1)/2 nontrivial frequencies
  "checkpoint_steps": [ …≈30 log-spaced… ],
  "norms": [ [57 values], … per checkpoint ], // ‖W_E projected on (cos_k, sin_k)‖, DC separate
  "key_freqs": [14, 35, 41, 42, 52]
}
```

### 4.5 `embedding_snapshots.json`

```jsonc
{
  "schema_version": "1.0",
  "d_model": 128, "p": 113,
  "snapshots": [
    { "step": 0,     "label": "init",       "matrix": { "scale": 0.0123, "data_b64": "…" } },
    { "step": 1400,  "label": "memorized",  "matrix": { … } },
    { "step": 5000,  "label": "forming",    "matrix": { … } },
    { "step": 11000, "label": "grokked",    "matrix": { … } },
    { "step": 40000, "label": "final",      "matrix": { … } }
  ]                                           // int8-quant, 113 rows × 128 cols, row-major
}
```

### 4.6 `clock.json`

```jsonc
{
  "schema_version": "1.0",
  "p": 113,
  "key_freqs": [14, 35, 41, 42, 52],
  "alpha": [ … 5 fitted coefficients … ],     // logit(c) ≈ Σ α_k cos(w_k (a+b−c)); least-squares fit
  "fit_r2": 0.0,                              // variance in centered logits explained by the fit
  "embed_angles": {                            // MEASURED angle of each token on each key-freq plane
    "14": [ 113 radians ], "35": [ … ], "41": [ … ], "42": [ … ], "52": [ … ]
  },
  "phase_offsets": { "14": 0.0, … }            // per-freq phase so ideal angle = w_k·a + offset
}
```

`InterferenceReadout` computes the 113 logit values client-side from `alpha` — pure trigonometry on measured coefficients, badge-labeled "fitted approximation (R² = …)".

### 4.7 `ablations.json`

```jsonc
{
  "schema_version": "1.0",
  "checkpoint_step": 40000,
  "eval_split": "val",
  "baseline": { "loss": 0.0, "acc": 0.0 },
  "presets": [
    { "id": "keep_key_only",  "loss": 0.0, "acc": 0.0 },   // ablate all 51 noise freqs
    { "id": "ablate_key",     "loss": 0.0, "acc": 0.0 },   // remove only the 5 key freqs
    { "id": "ablate_noise",   "loss": 0.0, "acc": 0.0 }    // same config as keep_key_only; kept
  ],                                                        //   as separate row for auditability
  "single_freq": [ { "k": 1, "loss": 0.0, "acc": 0.0 }, … 56 rows … ],   // ablate exactly one k
  "key_subsets": [                                           // all 2^5 kept-subsets × noise on/off
    { "kept_keys": [14,35,41,42,52], "noise": false, "loss": 0.0, "acc": 0.0 },
    { "kept_keys": [14,35,42],       "noise": true,  "loss": 0.0, "acc": 0.0 }
    // … 64 rows total
  ],
  "ablation_def": "zero the cos_k/sin_k components of the specified frequencies in the model's Fourier-space logit decomposition, then re-evaluate (matching Nanda et al.'s restricted/excluded methodology)"
}
```

### 4.8 `prediction_maps.json`

```jsonc
{
  "schema_version": "1.0",
  "p": 113,
  "train_mask": [ …rle-bitmask… ],
  "checkpoints": [
    { "step": 0,     "correct": [ …rle-bitmask… ] },
    { "step": 300,   "correct": [ … ] },
    { "step": 1400,  "correct": [ … ] },
    { "step": 5000,  "correct": [ … ] },
    { "step": 9400,  "correct": [ … ] },
    { "step": 10500, "correct": [ … ] },
    { "step": 11000, "correct": [ … ] },
    { "step": 40000, "correct": [ … ] }
  ]
}
```

### 4.9 `neurons.json`

```jsonc
{
  "schema_version": "1.0",
  "checkpoint_step": 40000,
  "grid_p": 113,
  "neurons": [
    { "id": 87, "dominant_freq": 42, "freq_purity": 0.94,
      "activations": { "scale": 0.02, "data_b64": "…" } }   // int8-quant 113×113, ReLU post-act
    // … 12 neurons, ≥2 per key frequency, chosen by purity
  ]
}
```

**Payload budget:** critical path (manifest + mainline + prediction_maps) ≲ 250 KB gzipped; everything else lazy per act. Nothing exceeds ~400 KB.

---

## 5. The scrollytelling script

Format per section: **[copy]** is on-page text (Spectral, on paper); *stage directions* describe the instrument. Numbers in `{braces}` are template slots filled from `manifest.json` at build time — the page never states a number the pipeline didn't measure. Internalize this section verbatim minus the braces; it is deliberately written to be spoken.

---

**S0 — Cold open.** *Instrument full-bleed. The train curve draws quickly to the top. Then the val trace crawls along the floor while the odometer spins — 100, 1,000, 10,000 steps — for a genuinely long four seconds (skippable; reduced-motion shows three stills). Then the cliff. Title fades in over the frozen chart.*

**[copy]** *Grokked.* A neural network memorized its training data in a few hundred steps — then sat at random-chance accuracy on everything else for ten thousand more. And then, long after any reasonable person would have stopped training, it snapped to perfect generalization. This is a true story, reconstructed from the checkpoints of a real model. Scroll to watch it happen — and then to find out why.

**S1 — The task.** *Instrument shows the 113×113 grid faintly; 30% of cells light amber (the train set), the rest stay dark.*

**[copy]** The task couldn't be smaller: add two numbers modulo 113 — clock arithmetic, `a + b`, wrap past 112 back to 0. There are exactly 113 × 113 = 12,769 possible questions. We show the network {n_train} of them — the amber cells — over and over, and never show it the rest. The only question that matters: will it *compute* the answers, or just *remember* them? For everything that follows, chance performance is 1-in-113 — about 0.9%.

**S2 — Memorization.** *Hero curve appears; train accuracy sweeps to 100% within the first few hundred steps; PredictionMap inset: amber-outlined cells flip green, the sea stays red.*

**[copy]** Memorizing takes almost no time. By step {memorization_end} the network is perfect on every pair it has seen — and still at chance on the pairs it hasn't. Look at the map: green confetti exactly where the training set is, red everywhere else. In the textbook telling, this is the end of the story. The model has overfit. Early-stopping lore says: you should have stopped already; nothing good happens from here.

**S3 — The plateau.** *Slow scroll-driven advance of `stepIdx` through the flat region. A bracket annotation stretches across the plateau. One beat calls out the axis itself.*

**[copy]** Then — nothing. For thousands of steps, validation accuracy is a flat line on the floor. And here you have to read the axis honestly: it's logarithmic. Every equal slice of this plateau is *ten times* more training than the slice before. The network re-reads the same {n_train} flashcards tens of millions of times. Loss on the training set is already near zero; by every visible sign, learning is over.

**S4 — The snap.** *Scroll releases the cliff: val trace leaps; PredictionMap floods green in two beats; odometer settles at {grok_step}.*

**[copy]** At step {grok_step}, the line leaves the floor — and within a short window the model is near-perfect on thousands of pairs it has *never seen*. Not perfect on some. Perfect on essentially all of them, simultaneously. Memorization can't do that; you can't recall what you never saw. Somewhere between the flashcards and the cliff, this network stopped remembering and started *computing*.

**S5 — The question.** *Everything freezes. The scrubber handle pulses once, inviting the reader to drag.*

**[copy]** It's like a student who crams the answer key: perfect on every practice problem, lost on anything new. Weeks pass. Nothing. Then one morning the *trick behind the problems* clicks, and suddenly every new problem is easy. The strange part isn't the click — it's how long after the cramming it came, and what that timing implies: something must have been changing the whole time, somewhere we weren't looking. Drag the timeline yourself. The outside gives nothing away. So we'll need two things: proof that this isn't a fluke — and a look inside.

---

**S6 — The data dial.** *Instrument switches to SweepCompare: curves for several training fractions fan out; smaller fractions grok later; the smallest never does within budget.*

**[copy]** First: it's lawful. Give the network a larger share of the 12,769 pairs and the plateau shortens. Shrink the share and the plateau stretches — not linearly, *violently*: below roughly {critical_frac_note}, the model doesn't grok at all within our {budget_steps}-step budget. That sharp sensitivity is the signature of a phase transition, not a glitch. There seems to be a minimum amount of evidence required for the "real rule" to be findable at all.

**S7 — Weight decay is the clock.** *Curves for wd ∈ {0, 0.1, 0.3, 1.0, 3.0} at fixed fraction; wd = 0 flatlines with a "no grok within budget" tag.*

**[copy]** Second lever: weight decay — a penalty on the size of the network's weights, a rent charged on every parameter, every step. Turn it up and grokking comes sooner. Turn it down and the plateau stretches. Turn it *off* and, within our budget, the snap never comes. Hold that thought, because it's the biggest clue we have: whatever eventually generalizes isn't just accurate — it must be *cheaper to represent* than memorization, and weight decay is what makes cheapness matter.

**S8 — The phase map.** *Full grid heatmap; color = log steps-to-grok; hatched DNF cells; the mainline cell marked ◦. Clicking any cell loads its curve into SweepCompare with the mainline ghost.*

**[copy]** Here's the whole landscape — every combination we trained, {n_runs} runs. Bright cells grok fast; dark cells grok slowly; hatched cells never grokked before the budget ran out. Two clean gradients, one toward more data, one toward more decay. A phenomenon this orderly isn't magic. It has a mechanism. Time to open the box.

---

**S9 — A change of basis.** *Instrument shows EmbeddingHeatmap at the final checkpoint: 113 × 128 static.*

**[copy]** Every number 0–112 enters the network as a learned list of 128 values — its *embedding*. Here are all 113 of them, stacked. It looks like noise, and in one sense it is: no individual value means anything. But "meaningless as a list" and "meaningless" are different claims. Ask a different question — for each possible frequency, *how much does this table oscillate at that rate as you count 0, 1, 2, … 112?* — and the noise resolves.

**S10 — Five frequencies.** *Heatmap slides aside; FourierSpectrum appears. The checkpoint slider (synced to the global step) plays: at init, grass; during the plateau, five violet spikes rise out of the floor; by the grok step they dominate.*

**[copy]** In frequency space the embedding isn't noise at all: {n_key} spikes stand up out of the floor — for this run, frequencies {key_freqs}. The network has learned to represent each number `a` by, essentially, `cos` and `sin` of `a` at a handful of chosen rates: points on a few circles. Now drag through training and watch *when* the spikes grow. They rise **during the plateau** — while validation accuracy is still flat on the floor. This is the hidden construction we inferred back at the cliff, caught on camera.

**S11 — Addition becomes rotation.** *ClockIntro: pick `(a, b)`; the hand rotates to `a`, advances by `b`, lands on the sum. Then the reveal: DialArray — five dials, angles taken from the actual learned embeddings, spinning at k× speed.*

**[copy]** Why circles? Because on a clock, you don't need arithmetic to add — you *rotate*. Eight o'clock plus six hours: turn the hand, read where it lands. Wrap-around comes free; that's the whole point of a dial. Modular addition *is* clock addition with 113 hours — and the network found this. It doesn't keep one clock, though. It keeps {n_key}, each ticking at its own rate — a frequency-{k_example} dial moves {k_example} notches per unit. Composing two rotations on each dial is done with the trigonometric identities for `cos(a+b)` and `sin(a+b)` — angle addition, implemented in weights. These dial positions aren't an artist's rendering, by the way: they're the model's actual learned embedding angles.

**S12 — The readout: interference.** *InterferenceReadout: one dial's cosine preference is a broad bump over candidate answers; stack dials one by one and the bump sharpens into a spike at `a+b mod 113`, ripples canceling elsewhere.*

**[copy]** One dial already points near the truth, but softly — a broad preference, easy to disturb. The network's final trick is to score every candidate answer `c` by summing, across its dials, `cos(w_k·(a + b − c))`. Each term peaks exactly when `c` is the true sum; away from it, the dials disagree and their votes cancel. Stack five frequencies and the broad bump collapses into a single sharp spike — constructive interference at the answer, destructive everywhere else. If that maneuver sounds familiar, it should: it's how a Fourier transform finds a signal. This curve is drawn from a five-coefficient formula fitted to the real model's outputs — and that tiny formula explains {logit_fit_r2} of them.

**S13 — Surgery.** *AblationBoard. The reader runs the three-beat sequence; MetricReadout swings against the ln(113) "coin-flip ceiling" line.*

**[copy]** If those five frequencies really are the algorithm, we can prove it with a knife. Every number below comes from actually re-running the model with pieces removed — nothing is simulated. **Cut everything except the five key frequencies:** accuracy {keep_key_acc}. Fifty-one of fifty-six frequencies gone, and the model doesn't care. **Cut only the five:** accuracy {ablate_key_acc} — the model collapses to roughly coin-flip loss. Five slots out of fifty-six were carrying everything. And the punchline — **cut only the fifty-one "noise" frequencies:** the model gets *better* than it was ({ablate_noise_loss} vs {baseline_loss} loss). The leftover mass wasn't neutral; it was the residue of the old memorization, still whispering wrong answers. The clean algorithm was in there, wrapped in scaffolding.

**S14 — Three phases, one movie.** *Return to the Hero. X-ray traces switch on: key-frequency energy, excluded loss, restricted loss, weight norm. Phase bands label memorization / circuit formation / cleanup. A guided scrub replays the whole run.*

**[copy]** Now rewatch the training run with x-ray vision, using the measures from Nanda et al. — and the single event becomes three overlapping phases. **Memorization** (to ~step {memorization_end}): train loss dives; the "excluded loss" — training loss with the key frequencies surgically removed — stays *low*, proving the right answers aren't flowing through those frequencies yet. **Circuit formation** (to ~step {formation_end}): key-frequency energy climbs steadily and excluded loss peels upward — the network is shifting its answers onto the circuit — *while validation accuracy stays flat*. Gradual inside, invisible outside. **Cleanup** (to ~step {cleanup_end}): weight decay strips the now-redundant memorization machinery; total weight norm drops, and *that* is when validation accuracy jumps. Read it again, because it's the reframe the whole piece was built for: **the sudden click you saw in Act I was mostly demolition, not construction.** The understanding was already there; what snapped was the removal of the noise around it.

**S15 — Why the circuit wins.** *Quiet section; a small two-pan balance graphic: "lookup table" pan piled high (weight norm) vs. "five dials" pan light, both achieving zero train loss.*

**[copy]** Which leaves the last why. Both solutions ace the training set, so gradient descent alone has no reason to prefer one. But they differ in *cost*. A lookup table is a mansion of special cases — thousands of independent facts, each stored in weight mass. The dial algorithm is a studio apartment: a few frequencies and three trig identities that answer *every* question. Weight decay charges rent on weight mass — so from the moment a circuit starts forming, the memorizing solution is bleeding, and the efficient one is compounding. Given enough steps, rent decides. That's also why the levers of Act II point the way they do: more data makes the general rule easier to find and memorization more expensive to maintain; more decay raises the rent. No rent, no eviction — no grok.

**S16 — Coda + Methods.** *Instrument dims to the frozen hero curve. Methods footer follows.*

**[copy]** One small transformer, one toy task, fully reverse-engineered — and it was doing *mathematics*: a Fourier representation, angle-addition identities, an interference readout. Nobody put that there. It condensed out of gradient descent and a rent bill. Hold the two morals in tension. First: a model whose visible behavior says "stuck" can be mid-construction inside — external metrics are a shadow of what's really happening, which is both a caution for anyone training models and the whole case for interpretability. Second: don't over-generalize a fable — this is one task, one small model; whether big models "grok" this cleanly is open research, and follow-up work (on efficiency races between circuits, and on where grokking does and doesn't appear) is linked below. Everything you just scrolled through was reconstructed from {n_checkpoints} checkpoints of a model we trained ourselves to the recipe of Nanda et al.; every number on this page came out of that pipeline, and the code for both halves is public.

*Methods footer: run card from `manifest.json`, citations (Power et al. 2022, arXiv:2201.02177; Nanda et al. 2023, arXiv:2301.05217), pipeline + site repo links, limitations note, keyboard-shortcut list.*

---

## 6. Build prompts for Claude Code

### 6(a) — Data pipeline prompt

Run in a Python environment with a GPU (Colab T4 is fine). Paste as-is:

```text
You are building the offline data pipeline for an interactive explainer about grokking
(Power et al. 2022, arXiv:2201.02177; Nanda et al. 2023, arXiv:2301.05217). Read
docs/BLUEPRINT.md §4 (the data contract) and §0/§7 first — §4 is the source of truth for
every output file. Work in pipeline/ per the tree in §2.

GOAL: train a 1-layer transformer on modular addition, reproduce grokking, extract the
Fourier-circuit interpretability data, and export JSON files that validate against §4.

PRIMARY PATH — train from scratch (preferred for clean provenance; it also lets the site
say "our reproduction"). Reference recipe (Nanda et al.'s setup):
  task: a + b mod p, p = 113; input tokens [a, b, '='], predict c at the final position;
        vocab 114 (numbers 0..112 plus '='), learned positional embeddings.
  model: 1 transformer layer, d_model=128, 4 heads, d_head=32, d_mlp=512, ReLU, NO LayerNorm.
  data:  30% of all 113^2 pairs as train (seeded split), rest val. FULL-batch training,
         so 1 epoch = 1 optimization step.
  optim: AdamW, lr 1e-3, betas (0.9, 0.98), weight_decay 1.0, 40_000 steps, seed 0.
  Expected shape: train acc ~100% within a few hundred steps; val acc near chance
  (~0.9%) for thousands of steps; grok (val ≥99%) somewhere in the ~5k–20k range.
  If the mainline run does not grok by 40k steps, STOP and report — do not fudge.
SECONDARY PATH (optional cross-check): Neel Nanda released the paper's checkpoints and an
annotated Colab (via github.com/neelnanda-io — locate the current link from his Grokking
paper page/README rather than hardcoding URLs). If reachable, verify our detected key
frequencies and ablation directions against his; note agreement in the manifest.

CHECKPOINTING / EVAL SCHEDULE:
  - Evaluate metrics on a log-spaced schedule, ~400 unique integer steps from 0 to 40k
    (dense early). At each eval: train/val loss+acc on the full splits.
  - Save full weight checkpoints at ~30 log-spaced steps (for spectrum-over-time) plus the
    5 snapshot steps used in embedding_snapshots.json and prediction_maps.json checkpoints.

ANALYSIS (fourier.py / analysis.py) — follow Nanda et al.'s definitions; consult their
public colab for the exact restricted/excluded-loss construction rather than improvising:
  1. Fourier basis over Z_113: DC + {cos, sin}(2πk·x/113) for k = 1..56, normalized.
  2. Per-checkpoint spectrum: project token embeddings W_E (number rows only) onto the
     basis; per-frequency norm = combined cos+sin component norm. -> spectrum_over_time.json
  3. Key-frequency detection: top-k per-frequency norms at the final checkpoint, k chosen
     at the largest gap (elbow) in the sorted norms; sanity-check k is small (3–8).
  4. Progress measures per eval step (mainline.json "progress"): key_freq_frac, gini over
     frequency norms, restricted_loss (keep only key-freq components of the Fourier-space
     logit decomposition), excluded_loss (remove key-freq components; evaluate on TRAIN),
     w_norm (sqrt of the sum of squares of all parameters).
  5. Phase boundaries (write to manifest, with the detection rule in a comment field):
     memorization_end = first step train_acc ≥ 0.99; formation_end = step where
     excluded-loss has risen a set fraction of its total rise / key_freq_frac crosses a
     threshold; cleanup_end = first step after grok_step where w_norm's decline flattens.
     These are heuristics — implement them as documented, deterministic rules.
  6. Ablations at the final checkpoint (ablations.json): implement frequency ablation by
     zeroing the chosen frequencies' components in the Fourier-space logit decomposition
     (Nanda's methodology), then re-evaluate on val. Produce: baseline; the 3 presets;
     all 56 single-frequency ablations; all 32 kept-subsets of key freqs × noise on/off
     (64 rows). Expected directions (verify, don't assume): keep-key-only ≈ baseline or
     better; ablate-key ≈ collapse toward uniform loss ln(113)≈4.727; ablate-noise
     slightly BETTER than baseline. Report if any direction disagrees.
  7. clock.json: per key frequency, the measured angle of each token on its (cos_k, sin_k)
     embedding plane (atan2), plus per-frequency phase offsets; fit logit(a,b,c) ≈
     Σ_k α_k cos(w_k(a+b−c)) by least squares over all (a,b) in val and all c; store
     alpha and fit R² (also copy R² into manifest.logit_fit_r2).
  8. prediction_maps.json: at the 8 listed checkpoint steps, argmax predictions for ALL
     12,769 pairs -> correctness bitmask, RLE-encoded per §4 conventions; plus train mask.
  9. neurons.json: final-checkpoint MLP post-ReLU activations over the full (a,b) grid;
     for each neuron compute its dominant frequency and purity (share of activation-map
     Fourier energy in that frequency); export the 12 highest-purity neurons covering
     ≥2 per key frequency, int8-quantized.

SWEEP (sweep.py):
  Default grid: fractions [0.15,0.2,0.25,0.3,0.4,0.5,0.7] × weight_decays [0,0.1,0.3,1,3],
  seed 0, cap 30_000 steps, early-stop once val_acc ≥ 0.99 sustained for 10 evals
  (record grok_step per the definition in §4.3). Slim eval schedule (~150 log-spaced
  points) -> runs/*.json. status ∈ {grokked, not_grokked, diverged}.
  Provide --quick mode: only the row wd=1.0 and the column frac=0.30 (the "cross"),
  and set sweep.json "mode":"cross". Print a wall-clock estimate before starting and
  after the first run completes; if full grid projects > 8 GPU-hours, recommend --quick.

EXPORT + VALIDATION:
  - export.py writes every file in §4 with schema_version "1.0"; implement int8-quant and
    rle-bitmask exactly as specified; fill manifest completely (including provenance and
    detected values — nothing hardcoded that the pipeline can measure).
  - validate_contract.py: jsonschema definitions mirroring §4; validate every emitted file
    and cross-check invariants (array lengths match steps; RLE sums = 12,769; ablation row
    counts 3/56/64; key_freqs consistent across files). CI-style nonzero exit on failure.
  - Also emit pipeline/report.md: the hero curve as a matplotlib PNG, detected key freqs,
    phase boundaries, grok step, ablation table, fit R², sweep summary, wall-clock — so a
    human can eyeball correctness before the frontend exists.

QUALITY BARS: deterministic given seeds; float32; resumable checkpointing; requirements.txt
pinned; every "expected direction" above is asserted with a clear failure message rather
than silently exported. Total public/data payload must stay under ~1.5 MB gzipped —
downsample/quantize as specified, never invent data.
```

### 6(b) — Frontend prompt

Run in the `web/` app root. Paste as-is:

```text
You are building "Grokked", a single-page interactive scrollytelling explainer about
grokking, as a portfolio-flagship for a frontend/AI-product engineering candidate. Read
docs/BLUEPRINT.md first: §1.3 (design language — follow it exactly, including palette
tokens, type stack, and motion principles), §3 (component tree and the zustand store
contract — implement the store exactly as specified, including the story/manual mode
rules), §4 (data contract — mirror it in zod; NEVER render a number that isn't in the
JSON), and §5 (the scrollytelling script — the on-page copy, verbatim, with {slots}
filled from manifest.json at build/load time).

STACK: Next.js (App Router) + TypeScript strict, Tailwind (tokens from §1.3 as CSS
variables), D3 for bespoke visuals, Framer Motion for orchestration, zustand for state,
a small custom IntersectionObserver hook for scrollytelling (no heavy scroll lib).
Static-exportable; deploys on Vercel. Data loads from /public/data with zod validation;
heavy files lazy-load per act with prefetch-on-approach; show skeleton states inside the
Instrument, never layout shift.

BUILD ORDER (commit at each stage; each stage must run):
  1. Scaffold + tokens + fonts (Spectral, IBM Plex Mono, Archivo via next/font) + data
     layer (schemas.ts, loaders.ts, rle.ts, quant.ts with unit tests for RLE/quant).
  2. VERTICAL SLICE: Instrument (bezel, Odometer, DataBadge) + HeroCurve with log-x axis,
     phase bands, working Scrubber, and transient-subscription rendering (no React
     re-render per scrub frame) + PredictionMap synced to stepIdx. Deployable and
     impressive on its own.
  3. ScrollStage + StepText choreography for Act I (S0–S5 from §5), including the
     story/manual mode machine: scroll drives stepIdx via section keyframes; user scrub
     switches to manual; section boundary resyncs. The S0 cold open per §5 (skippable,
     reduced-motion fallback to three stills).
  4. Act III core: FourierSpectrum + EmbeddingHeatmap + CheckpointSlider (S9–S10);
     ClockIntro, DialArray, InterferenceReadout with the (a,b) picker (S11–S12);
     AblationBoard + FreqToggleGrid + PresetRow + MetricReadout resolving ONLY to
     precomputed rows (S13); Hero x-ray mode + phase-band replay (S14–S15).
  5. Act II: PhaseMap (handle grid AND cross modes from sweep.json; hatched not_grokked
     cells) + SweepCompare with mainline ghost (S6–S8).
  6. Coda + MethodsFooter (run card rendered from manifest), SectionRail with keyboard
     1–9 section jumps, opengraph-image, favicon.
  7. Polish pass: micro-interactions, hover cross-highlighting via hoveredFreq
     (Spectrum ↔ DialArray ↔ AblationBoard), copy slots audit, empty/error states for
     every loader (honest "data missing" inside the Instrument, per §1.3's writing rules).

NON-NEGOTIABLES:
  - 60fps: all step-synced views (HeroCurve playhead, PredictionMap, Odometer, Spectrum)
    update via store.subscribe + requestAnimationFrame with direct canvas/DOM mutation;
    React re-renders only on section/run/toggle changes. Canvas for the p×p grids with
    devicePixelRatio scaling and pre-rendered offscreen buffers per checkpoint.
  - Accessibility: every control keyboard-operable (Scrubber is a real range input over
    eval indices; toggles are buttons with aria-pressed; dials' (a,b) picker uses number
    inputs); visible focus rings; sr-only text summaries for each chart describing the
    current state; prefers-reduced-motion swaps all autoplay/scroll animation for stepped
    crossfades with visible controls; color never the sole encoding (train/val differ in
    dash pattern; DNF cells hatched).
  - Honesty affordances: DataBadge on every visualization (variants per §3.1); the
    InterferenceReadout badge shows fit R² from clock.json; AblationBoard never
    interpolates — configurations resolve to real precomputed rows only.
  - Responsive: laptop-first (this will be screen-shared in interviews), sane down to
    ~768px (Instrument stacks above text; PhaseMap scrolls horizontally); no mobile
    heroics beyond that.
  - Performance budget: initial route JS < 180 KB gz; critical data < 250 KB gz;
    Lighthouse ≥ 95 across the board on the deployed build.
  - Never invent data values or copy beyond §5. If a JSON field is missing, fail loudly
    in dev and show the honest error state in prod.

Definition of done: `next build` clean with strict TS and zero eslint errors; RLE/quant
unit tests pass; keyboard-only walkthrough of the full story possible; a README with a
30-second local-run guide and a paragraph crediting both papers and the pipeline.
```

---

## 7. Scope & risk plan

**Build sequence (working vertical slice first).** Weekend 1: pipeline mainline run + `mainline.json` + `prediction_maps.json` + frontend stages 1–2 → deploy. Even alone, "real reconstructed run + scrubber + prediction map" is a portfolio piece. Week 2: Act I choreography + Act III core (spectrum, clock, ablation) — this is the depth payload. Week 3: sweep + Act II + x-ray replay + polish. Interview-readiness gate: S0–S5 and S9–S14 working beats everything else.

**Fragility ranking and mitigations.**

1. **Sweep compute** — biggest line item (~35 runs; with early stopping expect roughly 4–8 GPU-hours on a T4, less on anything modern). Mitigation: early stopping, 30k cap, `--quick` cross mode; the contract's `mode` field means the frontend handles either without rework.
2. **Mainline doesn't grok / detected phases look wrong** — low risk with the reference recipe, but the pipeline is instructed to *stop and report*, never fudge. Cross-check against Nanda's released checkpoints if reachable.
3. **Scroll/scrub state fights** — the classic scrollytelling failure. That's why the story/manual mode machine is specced in the store contract and built in stage 3, not bolted on.
4. **Payload bloat** — embedding snapshots are the risk; contract caps them at 5 checkpoints, int8. Budget asserted in the pipeline validator.
5. **Checkpoint hosting drift** — solved by making self-training the primary path; Nanda's artifacts are a cross-check, not a dependency.
6. **Canvas perf on Safari** — pre-rendered offscreen buffers + integer scaling; test early in stage 2.

**Cut order if time runs short** (each cut leaves a coherent piece): ① NeuronGallery → ② full sweep grid becomes the cross → ③ raw EmbeddingHeatmap (keep the spectrum — it carries the beat) → ④ ClockIntro single dial only (keep DialArray + InterferenceReadout — never cut the interference plot). **Never cut:** Hero + scrubber + x-ray mode, PredictionMap, Spectrum-over-time, Ablation presets. Those four are the recruiter demo *and* the depth proof.

**Stretch (only after everything above):** "interview mode" — `?` overlay listing shortcuts; keys 1–9 jump to story beats so you can drive the piece fluidly on a screen-share without scrolling.

---

## Appendix A — Interview fact sheet

**Cold facts to know by heart.** Task: `a + b mod 113`; 12,769 pairs; 30% train (~3,831); chance = 1/113 ≈ 0.9%; uniform loss ln 113 ≈ 4.73. Model: 1-layer transformer, d_model 128, 4 heads, d_mlp 512, ReLU, no LayerNorm; AdamW lr 1e-3, weight decay 1.0, full batch, 40k steps. Fourier basis on Z₁₁₃: DC + 56 cos/sin pairs. Paper 2's mainline run found 5 key frequencies (theirs: 14, 35, 41, 42, 52 — *seed-dependent; quote your own run's set*). Circuit: embed `a` as cos/sin at key freqs → trig angle-addition identities compose `a+b` → logits ≈ Σₖ αₖ cos(wₖ(a+b−c)), argmax at the true sum via constructive interference. Three phases: memorization → circuit formation → cleanup; the visible val-accuracy jump happens during **cleanup**.

**Six questions you will be asked, with the strong answer:**

1. *"Why does it generalize at all — what breaks the tie between memorizing and the algorithm?"* Both reach zero train loss, so the tie-breaker is weight decay: the Fourier circuit achieves the same loss with far less weight norm, so once it starts forming, regularized gradient descent steadily reallocates mass toward it. No decay (or other regularization pressure), no reliable grok in this setting.
2. *"Why is the transition sudden if the mechanism is gradual?"* Accuracy is a thresholded, winner-take-all readout of logits. The circuit's logit share grows smoothly; validation accuracy only moves when the circuit's vote starts beating the memorization noise at the argmax — and cleanup removes that noise quickly. Smooth latent progress, discontinuous visible metric.
3. *"Why multiple frequencies instead of one?"* One frequency already prefers the right answer but with a broad, shallow peak. Summing several incommensurate cosines sharpens the peak and cancels the rest — same reason a DFT localizes a signal. Empirically, ablating down to one key frequency degrades margin/accuracy; the shipped `key_subsets` table shows it.
4. *"How do you know the circuit story is causal, not just correlational?"* Ablation both ways: delete the 5 key frequencies → collapse toward uniform; delete the other 51 → performance *improves*. Plus the excluded-loss curve: early in training, removing key frequencies barely hurts train loss (answers don't route through them); later it destroys it.
5. *"What did you actually build vs. reuse?"* Trained the model myself to the paper's recipe, reimplemented the analyses (Fourier projections, progress measures, ablations) against the paper/colab definitions, verified directions against the released checkpoints, and designed/built the entire frontend and data contract. Nothing on the page is simulated live; every number traces to a checkpoint.
6. *"Does this explain grokking in general?"* No — and the piece says so. It fully explains this task/model; follow-up work studies when grokking appears elsewhere (e.g., loss-landscape and norm-based accounts, and circuit-efficiency framings of generalization vs. memorization races). The honest claim: a complete mechanistic account of one clean case, plus transferable measurement ideas.

**One-breath summary (memorize this):** *"It memorized first because that's easy. Underneath, weight decay's rent slowly favored a cheaper solution — a Fourier circuit that adds numbers by rotating on a few circles and reads out the answer by interference. The visible jump is the cleanup at the end: demolition of the memorization scaffolding around an algorithm that had already formed."*
