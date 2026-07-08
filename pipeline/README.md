# Pipeline

Offline data generation. Run in a GPU environment (Colab T4 is fine).

1. `pip install -r requirements.txt`
2. Paste BLUEPRINT §6a into Claude Code to build train.py / analysis.py / export.py etc.
3. Run the mainline first. Inspect `report.md` — you want the grokking curve.
4. Then run the sweep (`sweep.py`, or `sweep.py --quick` if GPU time is tight).
5. `python validate_contract.py` — every JSON must pass before the frontend uses it.

Outputs land in `../web/public/data/` (the data contract, BLUEPRINT §4).
Checkpoints (.pt) are gitignored.
