# Grokked

An interactive web explainer about **grokking** — the moment a neural network stops
memorizing and suddenly starts understanding.

Two halves:
- **`pipeline/`** — Python. Trains a small model offline, extracts the data, writes JSON.
  Never deployed. Run this FIRST.
- **`web/`** — Next.js app. Renders everything from the JSON. Deployed on Vercel.

The full architecture, data contract, and scrollytelling script live in
**`docs/BLUEPRINT.md`** — read that first. The two build prompts (§6 of the blueprint)
are meant to be pasted into Claude Code.

---

## Order of work (don't skip)

1. **Pipeline first.** In Google Colab (free GPU), run the data-pipeline prompt
   (BLUEPRINT §6a). Start with just the mainline run. Check `pipeline/report.md`:
   you want a flat val-accuracy line that jumps up late. That's grokking. If you see it,
   the data is good.
2. **Then the frontend, stage 2 only.** Run the frontend prompt (BLUEPRINT §6b) and stop
   after the vertical slice: hero curve + scrubber + prediction map. Deploy it.
3. **Then everything else**, in the build order from BLUEPRINT §7.

## Quick start

Pipeline:
```bash
cd pipeline
pip install -r requirements.txt
# then paste BLUEPRINT §6a into Claude Code, or build train.py yourself
```

Frontend:
```bash
cd web
npm install
npm run dev
```

## The two papers this is built on
- Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic
  Datasets" (2022), arXiv:2201.02177 — the phenomenon.
- Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability"
  (2023), arXiv:2301.05217 — the mechanism.

All numbers on the deployed site are precomputed from real checkpoints. Nothing is
simulated live.
