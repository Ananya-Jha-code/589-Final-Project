from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()

# Features (8x8 images flattened into vectors)
X = digits.data

# Labels
y = digits.target

# Original 8x8 images
images = digits.images

print(X.shape)       # (1797, 64)
print(images.shape)  # (1797, 8, 8)
print(y.shape)       # (1797,)