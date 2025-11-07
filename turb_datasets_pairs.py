from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import numpy as np

import json
import torch
from pathlib import Path

def load_data_supervised(*, dataset_path, batch_size):
    dataset_train = TurboSupervisedDatasetNpz(dataset_path, split='train')
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

    dataset_val = TurboSupervisedDatasetNpz(dataset_path, split='val')
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

    dataset_test = TurboSupervisedDatasetNpz(dataset_path, split='test')
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

    return train_loader, val_loader, test_loader

class TurboSupervisedDatasetNpz(IterableDataset):
    def __init__(self, root: str, class_cond: bool = False, transform=None, split="train"):
        super().__init__()

        self.root = Path(root)
        self.class_cond = class_cond
        self.transform = transform
        self.split = split

        with open("index_luis.json", "r") as f:
            index = json.load(f)
        self.npz_files = index[split]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            npz_files = self.npz_files
        else:
            npz_files = self.npz_files[worker_info.id :: worker_info.num_workers]

        for npzf in npz_files:
            p = npzf["path"]

            with np.load(self.root / p) as f:
                lr_fields = torch.from_numpy(f["lr"]).to(torch.float32)
                hr_fields = torch.from_numpy(f["hr"]).to(torch.float32)

            assert len(lr_fields) == len(hr_fields) == npzf["num"]

            for i in range(npzf["num"]):
                lr = lr_fields[i]
                hr = hr_fields[i]

                if self.transform:
                    lr = self.transform(lr)
                    hr = self.transform(hr)

                yield lr, hr  # (input, target)

            del lr_fields, hr_fields

    def __len__(self):
        return sum(npzf["num"] for npzf in self.npz_files)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.split}, {len(self.npz_files)} files, {len(self)} samples)"
