import json
import pandas as pd
import numpy as np

# Load the JSON data
with open("training_results.json", "r") as f:
    data = json.load(f)

# Extract and organize metrics: class_id -> metric -> list of values
metrics = {}
for fold in data["folds"]:
    for class_id, class_metrics in fold["per_class_metrics"].items():
        if class_id not in metrics:
            metrics[class_id] = {}
        for metric, value in class_metrics.items():
            metrics[class_id].setdefault(metric, []).append(value)

# Get all class IDs and metric names
class_ids = sorted(metrics.keys(), key=int)
metric_names = list(metrics[class_ids[0]].keys())

# Generate one LaTeX table per metric
for metric_name in metric_names:
    rows = {}
    for class_id in class_ids:
        values = metrics[class_id][metric_name]
        mean = np.mean(values)
        std = np.std(values)
        rows[class_id] = {"Model 1": f"{mean:.2f} ± {std:.3f}"}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "class"

    latex_table = df.to_latex(
        caption=f"{metric_name.upper()} per Class Across Models",
        label=f"tab:{metric_name}_by_model",
    )
    print(latex_table)
    print("\n" + "-" * 80 + "\n")
