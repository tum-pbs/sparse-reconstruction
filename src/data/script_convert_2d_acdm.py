# convert data from acdm to pbdl dataloader format
import numpy as np
import os
import h5py
from utils import print_h5py

data_dir = "/mnt/ssdraid/pbdl-datasets-local/2d_acdm_raw"
out_dir = "/mnt/ssdraid/pbdl-datasets-local/2d_acdm_new"

metadata = {
    "incompressible-wake-flow": {
        "PDE": "Navier-Stokes: Incompressible Cylinder Flow",
        "Dimension": 2,
        "Fields Scheme": "VVp",
        "Fields": ["Velocity X", "Velocity Y", "Pressure"],
        "Domain Extent": [4.0,2.0],
        "Resolution": [128,64],
        "Time Steps": 1001,
        "Dt": 0.05,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["inflow", "open", "wall", "wall"],
        "Constants": ["Reynolds Number"],
    },

    "transonic-cylinder-flow": {
        "PDE": "Navier-Stokes: Compressible Cylinder Flow",
        "Dimension": 2,
        "Fields Scheme": "VVdp",
        "Fields": ["Velocity X", "Velocity Y", "Density", "Pressure"],
        "Domain Extent": [12.0,6.0],
        "Resolution": [128,64],
        "Time Steps": 1001,
        "Dt": 0.002,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["inflow", "open", "open", "open"],
        "Reynolds Number": 10000,
        "Constants": ["Mach Number"],
    },

    "isotropic-turbulence": {
        "PDE": "Navier-Stokes: Isotropic Turbulence",
        "Dimension": 2,
        "Fields Scheme": "VVVp",
        "Fields": ["Velocity X", "Velocity Y", "Velocity Z", "Pressure"],
        "Domain Extent": [1.5707,0.7854],
        "Resolution": [128,64],
        "Time Steps": 1001,
        "Dt": 0.002,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["open", "open", "open", "open"],
        "Reynolds Number": 433,
        "Constants": ["Z Slice"],
    },
}



os.makedirs(out_dir, exist_ok=True)

for root, dirs, files in os.walk(data_dir):
    for file in sorted(files):
        if file.endswith(".npz"):
            npzfile = np.load(os.path.join(root, file))
            data = npzfile["data"]
            const = npzfile["constants"]
            name = os.path.splitext(file)[0]
            name = name.replace("_", "-")

            print(os.path.join(out_dir, name))
            print(data.shape)
            print(const.shape)

            with h5py.File(os.path.join(out_dir, name + ".hdf5"), "w") as h5py_file:
                group = h5py_file.create_group("sims", track_order=True)
                for key in metadata[name]:
                    group.attrs[key] = metadata[name][key]

                for s in range(data.shape[0]):
                    dataset = h5py_file.create_dataset("sims/sim%d" % (s), data=data[s])
                    for m in range(len(metadata[name]["Constants"])):
                        key = metadata[name]["Constants"][m]
                        dataset.attrs[key] = const[s, m]
                h5py_file.close()

            print_h5py(os.path.join(out_dir, name + ".hdf5"))
            print("\n\n")
