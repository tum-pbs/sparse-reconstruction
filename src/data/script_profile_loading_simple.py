
import time
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from pbdl.torch.dataset import Dataset as PBDLDataset

datasets = ["transonic-cylinder-flow", "isotropic-turbulence"]
path = "/mnt/ssdraid/pbdl-datasets-local/2d_acdm"
workers = [0,4,16]

for dataset in datasets:
    ds = PBDLDataset(dset_name=dataset, local_datasets_dir=path, time_steps=2)
    print(f"Dataset: {dataset}", "Samples: " + str(len(ds) // 64))

    start = time.perf_counter()
    arr = np.load("/mnt/ssdraid/pbdl-datasets-local/2d_acdm_raw/%s.npz" % (dataset))
    a = arr["data"][0:1]
    b = arr["constants"][0:1]
    end = time.perf_counter()
    time_delta = (end-start)/60
    print("Numpy Raw (single file), LINEAR, Worker: 0, Time: %1.3f min" % (time_delta))

    for worker in workers:
        dataloader = DataLoader(ds, batch_size=64, num_workers=worker)

        start = time.perf_counter()
        for i, result in enumerate(dataloader):
            a,b,c = result
        end = time.perf_counter()
        time_delta = (end-start)/60
        print("HDF5, LINEAR, Worker: %d, Time: %1.3f min" % (worker, time_delta))

    for worker in workers:
        dataloader = DataLoader(ds, batch_size=64, num_workers=worker, sampler=RandomSampler(ds))
    
        start = time.perf_counter()
        for i, result in enumerate(dataloader):
            a,b,c = result
        end = time.perf_counter()
        time_delta = (end-start)/60
        print("HDF5, RANDOM, Worker: %d, Time: %1.3f min" % (worker, time_delta))
    print()
    print()
