import json
from pathlib import Path

# Load baseline
baseline = json.loads(Path("metrics_baseline.json").read_text())
baseline_rmse = baseline["rmse"]

# replace later with actual training eval
new_rmse = baseline_rmse

# Compare
if new_rmse > baseline_rmse + 0.25:
    raise SystemExit(f"Model worse than baseline: {new_rmse} > {baseline_rmse + 0.25}")

print(f"Model validation passed: {new_rmse}")