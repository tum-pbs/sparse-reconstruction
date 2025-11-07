import time, os
from collections import defaultdict
import h5py
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42 # prevent type3 fonts in matplotlib output files
plt.rcParams["ps.fonttype"] = 42
from mpl_toolkits.axes_grid1 import ImageGrid

import seaborn as sns



### SETTINGS
# NOTE: add file names with corresponding values to defaultdict to specify custom settings for specific dataset files

#data_dir = "/mnt/ssdraid/pbdl-datasets-local/2d_apebench_generated"
data_dir = "/mnt/ssdraid/pbdl-datasets-local/2d_jhtdb_downloaded"
#data_dir = "/mnt/ssdraid/pbdl-datasets-local/2d_well_downloaded"

out_dir = "plots/data"
out_format = "pdf"

num_seqences = defaultdict(lambda: 2, {}) # number of plotted sequences
rollout_start_fraction = defaultdict(lambda: 0.0, {}) # temporal start of the sequence to plot
rollout_end_fraction = defaultdict(lambda: 1.0, {}) # temporal end of the sequence to plot
rollout_num_timesteps = defaultdict(lambda: 10, {}) # number of individual timesteps per plotted sequence
cmap_per_field_str = defaultdict(lambda: "cividis", { # colormap per field index
    "Velocity X": "magma",
    "Velocity Y": "magma",
    "Velocity Z": "magma",
    "Vorticity": "icefire",
    "Density": "plasma",
    "Concentration": "viridis",
    "Concentration A": "viridis",
    "Concentration B": "viridis",
})

np.random.seed(42)


def load_data_from_hdf5(data_dir:str, file:str):
    hdf5_path = os.path.join(data_dir, file)

    data = []
    fields = []

    with h5py.File(hdf5_path, "r") as f:
        num_sims = len(f["sims"].keys())
        num_fields = len(f["sims"].attrs["Fields"])

        sim_ids = np.random.choice(range(num_sims), num_seqences[file], replace=False)
        sim_ids = np.sort(sim_ids)
        sim_shape = f["sims/sim%d" % sim_ids[0]].shape
        rollout_start = int(sim_shape[0] * rollout_start_fraction[file])
        rollout_end = int(sim_shape[0] * rollout_end_fraction[file])
        rollout_stride = (rollout_end - rollout_start) // rollout_num_timesteps[file]
        if rollout_stride < 1:
            print("Warning: Simulation %d has less timesteps than requested" % sim_id)
            rollout_stride = 1

        field_ids = np.random.choice(range(num_fields), num_seqences[file], replace=num_seqences[file] > num_fields)
        field_ids = np.sort(field_ids)
        for (sim_id, field_id) in zip(sim_ids, field_ids):
            sequence = f["sims/sim%d" % sim_id][rollout_start:rollout_end:rollout_stride] # extract data from simulation
            sequence = sequence[:, field_id] # extract field

            data.append(sequence)
            fields.append(f["sims"].attrs["Fields"][field_id])

        # metadata
        pde = f["sims"].attrs["PDE"]
        time_steps = list(range(rollout_start, rollout_end, rollout_stride))
        f.close()

    data = np.stack(data, axis=0)
    data = np.transpose(data, (0, 1, 3, 2)) # swap spatial dimensions
    min_size = min(data.shape[2], data.shape[3]) # center crop to square
    offset_x = (data.shape[2] - min_size) // 2
    offset_y = (data.shape[3] - min_size) // 2
    data = data[:, :, offset_x:offset_x+min_size, offset_y:offset_y+min_size]

    return data, time_steps, fields, pde, sim_ids


def plot_data(data:np.array, time_steps:list, out_file:str, fields:list, pde:str, sim_ids:np.array):
    num_sims = data.shape[0]
    num_timesteps = data.shape[1]

    fig = plt.figure(figsize=(1.4*num_timesteps, 1.45*num_sims), dpi=150)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(num_sims, num_timesteps),
        axes_pad=0.05,
        share_all=True,
        cbar_location="right",
        cbar_mode="edge",
        direction="row",
        cbar_size="5%",
        cbar_pad=0.1,
    )
    for i in range(num_sims):
        vmin = np.min(data[i])
        vmax = np.max(data[i])

        for j in range(num_timesteps):
            ax = grid[i * num_timesteps + j]
            cmap = cmap_per_field_str[fields[i]]
            im = ax.imshow(data[i, j], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_axisbelow(True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("Simulation %d" % sim_ids[i])
            ax.set_xlabel("t=%d" % time_steps[j])

        cbar = ax.cax.colorbar(im)
        cbar.set_label("%s" % (fields[i]))

    pde = pde.replace("Navier-Stokes", "NS")
    if "Gray Scott" in pde:
        pde = "Gray Scott: %s" % (out_file.split("/")[-1].replace("gs_", "").replace("_test", "").split(".")[0].title())
    if "_test" in out_file.split("/")[-1] and not "2d_well" in data_dir.split("/")[-1]:
        pde += " (test)"

    grid[num_timesteps - 1].annotate("%s" % (pde), xy=(1, 1), xycoords="axes fraction", fontsize=12,
                                        xytext=(-5, -2), textcoords="offset points",
                                        ha="right", va="top",
                                        bbox=dict(facecolor="whitesmoke",
                                                edgecolor="darkslategray",
                                                boxstyle="round,pad=0.5",
                                                alpha=1.0))

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()



for file in sorted(os.listdir(data_dir)):
    if not file.endswith(".hdf5"):
        continue

    hdf5_path = os.path.join(data_dir, file)
    print(hdf5_path)

    data, time_steps, field_str, pde, sim_ids  = load_data_from_hdf5(data_dir, file)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, os.path.splitext(file)[0] + "." + out_format)

    plot_data(data, time_steps, out_file, field_str, pde, sim_ids)
