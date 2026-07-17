"""
Generate a SYNTHETIC CIFAR-10-format dataset so the assignment notebooks run
without downloading the real 170MB dataset.

Why: the real file lives on www.cs.toronto.edu, which is often unreachable.
This writes pickle batches in the EXACT format that cs231n/data_utils.load_CIFAR10
expects, so no notebook cell needs to change.

The data is structured (10 class clusters + noise), so KNN/predict_labels and
cross-validation still produce meaningful accuracy numbers. Accuracy will NOT
match the "~27%" expected on real CIFAR-10 -- that number is data-dependent.

To use real data later: delete this whole cifar-10-batches-py/ directory and
run get_datasets.sh (or download cifar-10-python.tar.gz yourself).
"""
import os
import pickle
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar-10-batches-py")
NUM_CLASSES = 10
PER_BATCH = 10000           # CIFAR-10 has 10000 images per batch
DIM = 3072                  # 32 * 32 * 3
SEED = 1337
LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def make_centers():
    rng = np.random.RandomState(SEED)
    # Each class gets a fixed random prototype in [0, 255].
    centers = rng.uniform(60, 200, size=(NUM_CLASSES, DIM)).astype(np.float64)
    # The raw prototypes are too far apart, so 1-NN gets ~100% for every k.
    # Blending each class toward the global mean makes classes overlap a bit,
    # which lowers accuracy somewhat -- but note a fully realistic CIFAR-style
    # k-curve (~27% at k=1) is NOT reproducible from cluster+noise data, because
    # with ~500 same-class points the nearest neighbour is always very close.
    # Accuracy on this synthetic data stays high (~0.99); that is expected.
    global_mean = centers.mean(0, keepdims=True)
    centers = ALPHA * centers + (1.0 - ALPHA) * global_mean
    return centers


# Blend factor: fraction of the per-class signal kept (rest = global mean).
ALPHA = 0.3


def make_batch(centers, start_label, noise_std=60.0, seed=None):
    rng = np.random.RandomState(seed)
    labels = np.arange(start_label, start_label + PER_BATCH) % NUM_CLASSES
    data = centers[labels] + rng.normal(0.0, noise_std, size=(PER_BATCH, DIM))
    data = np.clip(np.round(data), 0, 255).astype(np.uint8)
    return {
        "data": data,
        "labels": labels.tolist(),
        "batch_label": "synthetic batch",
        "filenames": [f"synthetic_{i}.png" for i in range(PER_BATCH)],
    }


def write_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=2)  # protocol 2 -> loads fine with encoding="latin1"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    centers = make_centers()
    s = 1000  # seed offset per batch for reproducibility
    for b in range(1, 6):
        batch = make_batch(centers, start_label=(b - 1) * PER_BATCH, seed=s + b)
        write_pickle(os.path.join(OUT_DIR, f"data_batch_{b}"), batch)
        print(f"wrote data_batch_{b}")
    test_batch = make_batch(centers, start_label=0, seed=s + 99)
    write_pickle(os.path.join(OUT_DIR, "test_batch"), test_batch)
    print("wrote test_batch")
    meta = {
        "num_cases_per_batch": PER_BATCH,
        "label_names": LABEL_NAMES,
        "num_vis": DIM,
    }
    write_pickle(os.path.join(OUT_DIR, "batches.meta"), meta)
    print("wrote batches.meta")
    print(f"\nDone. Synthetic CIFAR-10 written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
