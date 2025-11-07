import numpy as np
import os, json, time
import h5py

from render import render_trajectory


#dataset_name ="channel"
#dataset_name ="isotropic1024coarse"
dataset_name ="mhd1024"
#dataset_name ="transition_bl"

path = "/mnt/ssdraid/pbdl-datasets-local/3d_jhtdb_downloaded/" + dataset_name
outpath = "/home/holzschuh/renders/" + dataset_name
start = 0
end = 4
step = 1

data = []
with h5py.File(path + ".hdf5", "r") as h5py_file:
    for t in range(start, end, step):
        data.append(h5py_file["sims/sim0"][t])
    h5py_file.close()

if dataset_name == "mhd1024":
    vmin, vmax = -0.7, 0.7
elif dataset_name == "isotropic1024coarse":
    vmin, vmax = -1.2, 1.2
elif dataset_name == "channel":
    vmin, vmax = None, None
else:
    vmin, vmax = None, None

render_trajectory(
    data=data,
    dimension=3,
    output_path=outpath,
    sim_id=0,
    time_steps=len(data),
    steps_plot=len(data),
    vmin=vmin,
    vmax=vmax,
    #vmin=-1,
    #vmax=1,
)