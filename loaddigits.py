import csv
from pathlib import Path

from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()

# Features (8x8 images flattened into vectors)
X = digits.data

# Labels
y = digits.target

# Original 8x8 images
images = digits.images

out_path = Path(__file__).resolve().parent / "digits.csv"
header = [f"pixel_{i}" for i in range(X.shape[1])] + ["label"]
with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row, label in zip(X, y):
        writer.writerow(row.tolist() + [int(label)])

print(X.shape)       # (1797, 64)
print(images.shape)  # (1797, 8, 8)
print(y.shape)       # (1797,)
print(f"Saved: {out_path}")