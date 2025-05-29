import json
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

# Paths to your JSON files
json_files = [
    "2021_classified.json",
    "2022_classified.json",
    "2023_classified.json",
    # "2024_classified.json"
]

# Data storage
all_data = []

# Load and parse JSON
for file in json_files:
    year = file[:4]
    with open(file, 'r') as f:
        data = json.load(f)
        for path, label in data.items():
            if label is None:
                label = 0  # Corrupted image => class 0
            try:
                parts = path.strip("/").split("/")
                date_str = parts[-2][:8]  # YYYYMMDD
                time_str = parts[-2][9:15]  # HHMMSS
                date = datetime.strptime(date_str, "%Y%m%d").date()
                time = datetime.strptime(time_str, "%H%M%S").time()
                time_decimal = time.hour + time.minute/60 + time.second/3600
                # Pseudo date: same year for all, just month-day for seasonal alignment
                pseudo_date = datetime(2000, date.month, date.day).date()
                all_data.append({
                    "year": int(year),
                    "date": date,
                    "time_decimal": time_decimal,
                    "label": int(label),
                    "pseudo_date": pseudo_date
                })
            except Exception as e:
                print(f"Skipping {path}: {e}")

# Convert to DataFrame
df = pd.DataFrame(all_data)
df["pseudo_date"] = pd.to_datetime(df["pseudo_date"])



# -------- PLOT 1: Total Images Per Pseudo Date --------
df_grouped = df.groupby(["pseudo_date", "year"]).agg(total_images=("label", "count")).reset_index()
pivot_total = df_grouped.pivot(index="pseudo_date", columns="year", values="total_images")

pivot_total.plot(title="Total Images per Field Season (July–October)", figsize=(10, 5))
plt.xlabel("Field Season Date (Month-Day)")
plt.ylabel("Total Images")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- PLOT 2: Average Imaging Time --------
df_grouped = df.groupby(["pseudo_date", "year"]).agg(avg_time=("time_decimal", "mean")).reset_index()
pivot_time = df_grouped.pivot(index="pseudo_date", columns="year", values="avg_time")

pivot_time.plot(title="Average Imaging Time per Field Season", figsize=(10, 5))
plt.xlabel("Field Season Date (Month-Day)")
plt.ylabel("Time of Day (Decimal Hours)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- PLOT 3: Class 1 Image Count --------
df_class1 = df[df["label"] == 1]
df_grouped = df_class1.groupby(["pseudo_date", "year"]).size().reset_index(name="class1_count")
pivot_class1 = df_grouped.pivot(index="pseudo_date", columns="year", values="class1_count")

pivot_class1.plot(title="Measurable (Class 1) Images per Field Season", figsize=(10, 5))
plt.xlabel("Field Season Date (Month-Day)")
plt.ylabel("Measurable Images")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- SUMMARY: Usable Image Proportion per Year --------
summary = df.groupby("year")["label"].agg(
    total="count",
    class1=lambda x: (x == 1).sum()
)
summary["proportion_class1"] = summary["class1"] / summary["total"]

print("\nProportion of Measurable (Class 1) Images per Year:\n")
print(summary)