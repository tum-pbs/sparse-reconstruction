import h5py
import numpy
import os


#path = "/mnt/ssdraid/pbdl-datasets-local/2d_acdm/isotropic-turbulence.hdf5"
#path = "/mnt/ssdraid/pbdl-datasets-local/2d_apebench_scraped/2d-phy-decay-turb-test.hdf5"
#path = "/mnt/ssdraid/pbdl-datasets-local/2d_apebench_generated/kolm_flow.hdf5"
#path = "/mnt/ssdraid/pbdl-datasets-local/3d_apebench_generated/gs_theta.hdf5"
path = "/mnt/ssdraid/pbdl-datasets-local/3d_jhtdb_downloaded/transition_bl.hdf5"
#path = "/mnt/ssdraid/pbdl-datasets-local/2d_well_downloaded/rayleigh_benard_train.hdf5"


with h5py.File(path, "r+") as f:
    print(type(f["sims"].attrs["PDE"]), f["sims"].attrs["PDE"])

    f["sims"].attrs["Boundary Conditions Order"] = ["x negative", "x positive", "y negative", "y positive", "z negative", "z positive"]
    f["sims"].attrs["Boundary Conditions"] = ["open", "open", "wall", "open", "open", "open"],

    print(type(f["sims"].attrs["Boundary Conditions Order"]), f["sims"].attrs["Boundary Conditions Order"])
    print(type(f["sims"].attrs["Boundary Conditions"]), f["sims"].attrs["Boundary Conditions"])
    f.close()


'''
path = "/mnt/ssdraid/pbdl-datasets-local/3d_apebench_generated/"

for file in os.listdir(path):
    if file.endswith(".hdf5"):
        hdf5_path = os.path.join(path, file)
        print(hdf5_path)

        with h5py.File(hdf5_path, "r+") as f:
            print(type(f["sims"].attrs["PDE"]), f["sims"].attrs["PDE"])

            f["sims"].attrs["Boundary Conditions Order"] = ["x negative", "x positive", "y negative", "y positive", "z negative", "z positive"]
            f["sims"].attrs["Boundary Conditions"] = ["periodic", "periodic", "periodic", "periodic", "periodic", "periodic"]

            print(type(f["sims"].attrs["Boundary Conditions Order"]), f["sims"].attrs["Boundary Conditions Order"])
            print(type(f["sims"].attrs["Boundary Conditions"]), f["sims"].attrs["Boundary Conditions"])
            f.close()
'''
