from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import numpy as np
import torch.distributed as dist

import json
import torch
from pathlib import Path

def load_data(*, dataset_path, batch_size, class_cond=False):
    
    dataset_train = TurboDatasetNpz(dataset_path, class_cond=class_cond, split='train')
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    
    dataset_val = TurboDatasetNpz(dataset_path, class_cond=class_cond, split='val')
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    
    dataset_test = TurboDatasetNpz(dataset_path, class_cond=class_cond, split='test')
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

    return train_loader, val_loader, test_loader

class TurboDatasetNpz(IterableDataset):
    def __init__(self, root: str, class_cond: bool, transform=None, split="train"):
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
            # single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Total number of processes * workers
        total_workers = world_size * num_workers

        # Calculate global worker id for correct sharding
        global_worker_id = rank * num_workers + worker_id

        # Shard npz_files by total workers, each gets every total_workers-th file starting from its global_worker_id
        npz_files = self.npz_files[global_worker_id :: total_workers]

        out_dict = {}
        if self.class_cond:
            raise NotImplementedError()

        for npzf in npz_files:
            p = npzf["path"]

            with np.load(self.root / p) as f:
                fields = torch.from_numpy(f["samples"]).to(torch.float32)

            assert len(fields) == npzf["num"]

            idxs = np.arange(npzf["num"])

            for i in idxs:
                field = fields[i]
                if self.transform:
                    field = self.transform(field)
                yield field, out_dict

            del fields
            
        #print(f"Rank {rank}, worker {worker_id}/{num_workers}, global_worker_id {global_worker_id}, files: {[f['path'] for f in npz_files]}", flush=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.split}, {len(self.npz_files)} files, {len(self)} fields)"

    def __len__(self):
        return sum(npzf["num"] for npzf in self.npz_files)