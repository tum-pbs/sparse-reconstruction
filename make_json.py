import json
import random

def generate_json(n, output_file="index_luis.json"):

    # Create and shuffle indices
    indices = list(range(n))
    random.shuffle(indices)

    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    data = {
        "train": [
            {
                "path": f"Ret180_192x65x192_fluct_c3_{i}.npz",
                "num": 20
            }
            for i in train_indices
        ],
        "val": [
            {
                "path": f"Ret180_192x65x192_fluct_c3_{i}.npz",
                "num": 20
            }
            for i in val_indices
        ],
        "test": [
            {
                "path": f"Ret180_192x65x192_fluct_c3_{i}.npz",
                "num": 20
            }
            for i in test_indices
        ]
    }

    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Generated {n} entries and wrote to {output_file}")

n = 300
generate_json(n)
