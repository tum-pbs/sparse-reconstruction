import h5py
import os
import numpy as np
from itertools import groupby
from utils import print_h5py


path = "/mnt/ssdraid/pbdl-datasets-local/3d_jhtdb_downloaded/channel.hdf5"
#path = "/mnt/ssdraid/pbdl-datasets-local/3d_jhtdb_downloaded/transition_bl.hdf5"

with h5py.File(path, "r+") as dset:
    print(type(dset["sims"].attrs["PDE"]), dset["sims"].attrs["PDE"])

    attrs = dset["sims/sim0"].attrs

    sim_shape = dset["sims/sim0"].shape
    meta = {
        "fields_scheme": dset["sims"].attrs["Fields Scheme"],
        "num_sims": len(dset["sims"]),
        "num_const": len(dset["sims"].attrs["Constants"]),
        "sim_shape": sim_shape,
        "num_frames": sim_shape[0],
        "num_sca_fields": sim_shape[1],
        "num_fields": len(list(groupby(dset["sims"].attrs["Fields Scheme"]))),
        "num_spatial_dim": len(sim_shape) - 2,
    }
    print(meta)


    num_sca_fields = meta["num_sca_fields"]

    # calculate the starting indices for fields
    field_indices = [0]
    for _, fe in groupby(meta["fields_scheme"]):
        field_indices.append(field_indices[-1] + len(list(fe)))

    # slim means that for vector (non-scalar) fields the std must first be broadcasted to the original size
    fields_std_slim = [0] * meta["num_fields"]

    fields_sca_std = np.full(  (num_sca_fields,) + (1,) * meta["num_spatial_dim"], 0 )
    fields_sca_mean = np.full( (num_sca_fields,) + (1,) * meta["num_spatial_dim"], 0 )
    fields_sca_min = np.full(  (num_sca_fields,) + (1,) * meta["num_spatial_dim"], float("inf") )
    fields_sca_max = np.full(  (num_sca_fields,) + (1,) * meta["num_spatial_dim"], -float("inf") )
    const_stacked = []

    # sequential loading of sims, norm data will be combined in the end
    for s in dset["sims/"]:
        sim = dset["sims/" + s]

        sub_attrs = dset["sims/sim" + str(int(s[3:]))].attrs
        const_stacked.append([sub_attrs[key] for key in dset["sims/"].attrs["Constants"]])

        axis = tuple(range(2, 2 + meta["num_spatial_dim"]))
        print(sim.shape)

        for t in range(meta["num_frames"]):
            print(t)
            fields_sca_std = np.add(
                fields_sca_std, np.std(sim[t:t+1], axis=axis, keepdims=True)[0]
            )
            fields_sca_mean = np.add(
                fields_sca_mean, np.mean(sim[t:t+1], axis=axis, keepdims=True)[0]
            )

            fields_sca_min = np.minimum(
                fields_sca_min, np.min(sim[t:t+1], axis=axis, keepdims=True)[0]
            )
            fields_sca_max = np.maximum(
                fields_sca_max, np.max(sim[t:t+1], axis=axis, keepdims=True)[0]
            )

            for f in range(meta["num_fields"]):
                field = sim[t:t+1, field_indices[f] : field_indices[f + 1], ...]

                # vector norm
                field_norm = np.linalg.norm(field, axis=1, keepdims=True)

                # frame dim + spatial dims
                axis = (0,) + tuple(range(2, 2 + meta["num_spatial_dim"]))

                # std over frame dim and spatial dims
                fields_std_slim[f] += np.std(field_norm, axis=axis, keepdims=True)[0]

    fields_sca_mean = np.array(fields_sca_mean) / (meta["num_frames"] * meta["num_sims"])
    fields_sca_std = np.array(fields_sca_std) / (meta["num_frames"] * meta["num_sims"])

    fields_std = []
    for f in range(meta["num_fields"]):
        field_std_avg = fields_std_slim[f] / (meta["num_frames"] * meta["num_sims"])
        field_len = field_indices[f + 1] - field_indices[f]
        fields_std.append(
            np.broadcast_to(  # broadcast to original field dims
                field_std_avg,
                (field_len,) + (1,) * meta["num_spatial_dim"],
            )
        )
    fields_std = np.concatenate(fields_std, axis=0)

    # caching norm data
    if "norm_fields_sca_mean" in dset:
        dset.__delitem__("norm_fields_sca_mean")
    if "norm_fields_sca_std" in dset:
        dset.__delitem__("norm_fields_sca_std")
    if "norm_fields_std" in dset:
        dset.__delitem__("norm_fields_std")
    if "norm_fields_sca_min" in dset:
        dset.__delitem__("norm_fields_sca_min")
    if "norm_fields_sca_max" in dset:
        dset.__delitem__("norm_fields_sca_max")
    if "norm_const_mean" in dset:
        dset.__delitem__("norm_const_mean")
    if "norm_const_std" in dset:
        dset.__delitem__("norm_const_std")
    if "norm_const_min" in dset:
        dset.__delitem__("norm_const_min")
    if "norm_const_max" in dset:
        dset.__delitem__("norm_const_max")

    dset["norm_fields_sca_mean"] = fields_sca_mean
    dset["norm_fields_sca_std"] = fields_sca_std
    dset["norm_fields_std"] = fields_std
    dset["norm_fields_sca_min"] = fields_sca_min
    dset["norm_fields_sca_max"] = fields_sca_max
    dset["norm_const_mean"] = np.mean(const_stacked, axis=0, keepdims=False)
    dset["norm_const_std"] = np.std(const_stacked, axis=0, keepdims=False)
    dset["norm_const_min"] = np.min(const_stacked, axis=0, keepdims=False)
    dset["norm_const_max"] = np.max(const_stacked, axis=0, keepdims=False)

