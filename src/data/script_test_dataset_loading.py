from pbdl.torch.dataset import Dataset as PBDLDataset
import torch
from torch.utils.data import DataLoader
import time, os
import h5py
import numpy as np
from utils import print_h5py



#path = "/mnt/ssdraid/pbdl-datasets-local/2d_acdm"
#path = "/mnt/ssdraid/pbdl-datasets-local/2d_apebench_scraped"
#path = "/mnt/ssdraid/pbdl-datasets-local/2d_apebench_generated"
#path = "/mnt/ssdraid/pbdl-datasets-local/2d_jhtdb_downloaded"
path = "/mnt/ssdraid/pbdl-datasets-local/2d_well_downloaded"
#path = "/mnt/ssdraid/pbdl-datasets-local/3d_apebench_generated"
#path = "/mnt/ssdraid/pbdl-datasets-local/3d_jhtdb_downloaded"
iterate = False

for file in sorted(os.listdir(path)):
    if file.endswith(".hdf5"):
        hdf5_path = os.path.join(path, file)
        print(hdf5_path)
        #print_h5py(hdf5_path)

        pbdl_all = PBDLDataset(dset_name=os.path.splitext(file)[0],
                            local_datasets_dir=path,
                            time_steps=1,
                            intermediate_time_steps=True,
                            normalize_const="mean-std",
                            normalize_data="mean-std")
                            #normalize_const=False,  # IMPORTANT: normalize...="mean-std" tries to compute norm buffers
                            #normalize_data=False,)  # which can causes memory issues for large 3D datasets, if buffers are
                                                    # not precomputed and exist in the dataset file

        # test if some metadata fields exist
        with h5py.File(hdf5_path, "r") as f:
            at = f["sims"].attrs
            print(type(at["PDE"]), at["PDE"])
            print(type(at["Dimension"]), at["Dimension"])
            print(type(at["Fields"]), at["Fields"])
            print(type(at["Fields Scheme"]), at["Fields Scheme"])
            #print(type(at["Domain Extent"]), at["Domain Extent"]) # NOTE: simulations with varied domain extent do not need a domain extent attribute in the main group
            print(type(at["Resolution"]), at["Resolution"])
            print(type(at["Time Steps"]), at["Time Steps"])
            print(type(at["Dt"]), at["Dt"])
            print(type(at["Boundary Conditions Order"]), at["Boundary Conditions Order"])
            print(type(at["Boundary Conditions"]), at["Boundary Conditions"])
            print(type(at["Constants"]), at["Constants"])
            f.close()

        print(len(pbdl_all))

        if iterate:
            # iterate over datasets with pytorch data loader
            start = time.perf_counter()
            loader = DataLoader(pbdl_all, batch_size=16, shuffle=False, num_workers=16)
            for i, data in enumerate(loader):
                d = data[0].shape
                if i % 100 == 0:
                    print(f"Batch {i}")
            print(f"Time: {(time.perf_counter() - start)/60} min")

        print("\n\n\n")
