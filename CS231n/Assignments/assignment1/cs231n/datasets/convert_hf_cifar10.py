"""Convert HuggingFace uoft-cs/cifar10 parquet -> CS231n cifar-10-batches-py format.

Output is byte-compatible in LAYOUT with the official cifar-10-python.tar.gz:
each batch is a pickle (protocol 2) with keys data (N,3072) uint8 [R1024,G1024,B1024
row-major], labels (list[int]), batch_label, filenames. Read back by
cs231n/data_utils.load_CIFAR10 with encoding="latin1".
"""
import io
import os
import pickle
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

OUT_DIR = "/home/yifeng/projects/Summer2026/The-Summer-of-2026/CS231n/Assignments/assignment1/cs231n/datasets/cifar-10-batches-py"
LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
DIM = 3072  # 32*32*3


def rows_to_array(parquet_path):
    """Read parquet, decode PNGs, return (data uint8 (N,3072), labels list[int])."""
    t = pq.read_table(parquet_path, columns=["img", "label"])
    n = t.num_rows
    bytes_col = t.column("img").combine_chunks().field("bytes").to_pylist()
    labels = t.column("label").to_pylist()
    data = np.empty((n, DIM), dtype=np.uint8)
    for i, b in enumerate(bytes_col):
        img = Image.open(io.BytesIO(b)).convert("RGB")
        if img.size != (32, 32):
            img = img.resize((32, 32))
        arr = np.asarray(img, dtype=np.uint8)          # (32,32,3) HWC
        data[i] = arr.transpose(2, 0, 1).reshape(DIM)   # -> [R,G,B] row-major
        if (i + 1) % 10000 == 0:
            print(f"  decoded {i+1}/{n}")
    return data, labels


def write_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=2)


def make_batch(data, labels, batch_label):
    return {
        "data": data,
        "labels": list(labels),
        "batch_label": batch_label,
        "filenames": [f"img_{i}.png" for i in range(len(labels))],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Reading train parquet (50000)...")
    train_data, train_labels = rows_to_array("/tmp/cifar_train.parquet")
    assert train_data.shape == (50000, DIM), train_data.shape
    print("Reading test parquet (10000)...")
    test_data, test_labels = rows_to_array("/tmp/cifar_test.parquet")
    assert test_data.shape == (10000, DIM), test_data.shape

    # Split train into 5 batches of 10000 (preserve original order, like official dataset)
    for b in range(5):
        s = b * 10000
        batch = make_batch(train_data[s:s+10000], train_labels[s:s+10000],
                           f"training batch {b+1} of 5")
        write_pickle(os.path.join(OUT_DIR, f"data_batch_{b+1}"), batch)
        print(f"wrote data_batch_{b+1}")

    write_pickle(os.path.join(OUT_DIR, "test_batch"),
                 make_batch(test_data, test_labels, "testing batch 1 of 1"))
    print("wrote test_batch")

    write_pickle(os.path.join(OUT_DIR, "batches.meta"), {
        "num_cases_per_batch": 10000,
        "label_names": LABEL_NAMES,
        "num_vis": DIM,
    })
    print("wrote batches.meta")
    print(f"\nDone. Real CIFAR-10 written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
