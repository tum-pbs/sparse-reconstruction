import numpy as np
import os, json, time
import multiprocessing as mp
import argparse
import h5py
import traceback

from pyJHTDB import libJHTDB
from tqdm import tqdm

from download_setups_3d import get_setup
from render import render_trajectory



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 3D PDE simulation data")
    parser.add_argument("--dataset_name", type=str, default="transition_bl")
    parser.add_argument("--out_name", type=str, default="transition_bl")
    parser.add_argument("--out_path", type=str, default="/mnt/ssdraid/pbdl-datasets-local/3d_jhtdb_new")
    parser.add_argument("--token", type=str, default="de.tum.georg.kohl-75077759") # TODO: remove token before publication

    args = parser.parse_args()



def prepare_data_and_log(json_path_log, h5py_path, p):
    # create log file for this set of simulations
    with open(json_path_log, "w") as f:
        newDict = {"all": p}
        json.dump(newDict, f, indent=4)
        f.close()

    # create h5py file for dataset or overwrite existing parameter
    with h5py.File(h5py_path, "a") as h5py_file:
        if not "sims" in h5py_file:
            dataset = h5py_file.create_group("sims", track_order=True)
        else:
            dataset = h5py_file["sims"]

        for key in p:
            dataset.attrs[key] = p[key]

        if not "sims/sim0" in h5py_file:
            time_steps = 1 + ((p["Temporal End"] - p["Temporal Start"]) // p["Temporal Step"])
            channels = len(p["Fields"])
            res_x = 1 + ((p["Spatial End"][0] - p["Spatial Start"][0]) // p["Spatial Step"][0])
            res_y = 1 + ((p["Spatial End"][1] - p["Spatial Start"][1]) // p["Spatial Step"][1])
            res_z = 1 + ((p["Spatial End"][2] - p["Spatial Start"][2]) // p["Spatial Step"][2])
            print("Created empty dataset with shape (%d, %d, %d, %d, %d)" % (time_steps, channels, res_x, res_y, res_z))
            for i in range(time_steps):
                print(f"sims/sim0/{i}")
                h5py_file.create_dataset(f"sims/sim0/{i}", shape=(1, channels, res_x, res_y, res_z), dtype=np.float32)

        h5py_file.close()


def core_download(token, data_set, field_code, t, start, end, step):
    lJHTDB = libJHTDB(token)
    lJHTDB.initialize(exit_on_error=False)

    result = []

    for field in field_code:
        downloaded = lJHTDB.getCutout(
            data_set = data_set,
            field = field,
            time_step = t,
            start = start,
            end = end,
            step = step,
            filter_width=1,
            )
        result += [downloaded]

    lJHTDB.finalize()

    result = np.concatenate(result, axis=-1)

    if start[0] % 50 == 0:
        print("Downloaded %s slice %04d at time step %d with shape %s" % (data_set, start[0], t, str(result.shape)))
    return (start[0], result)



def wrapped_download(token, data_set, field_code, t, start, end, step):
    try:
        return core_download(token, data_set, field_code, t, start, end, step)
    except:
        print("\n\n\n\n\n%" % (traceback.format_exc()))


def download_data(
        dataset_name: str,
        out_name: str,
        out_path: str,
        token: str,
    ):

    download_tries = 10
    timeout_min = 40 if dataset_name == "mhd1024" else 20
    poll_interval_min = 0.5
    workers = 10


    # create directories, paths, and logs
    out_dir = os.path.join(out_path, out_name)
    out_dir = out_dir[:-1] if out_dir[-1] == "/" else out_dir
    os.makedirs(out_dir, exist_ok=True)

    json_path_log = os.path.join(out_path, out_name + ".json")
    h5py_path = out_dir + ".hdf5"
    dataset, p, field_code = get_setup(dataset_name)
    prepare_data_and_log(json_path_log, h5py_path, p)

    start = np.array( p["Spatial Start"] ).astype(np.int32)
    end = np.array( p["Spatial End"] ).astype(np.int32)
    step = np.array( p["Spatial Step"] ).astype(np.int32)

    # main download
    data = []
    for t in tqdm(range(p["Temporal Start"], p["Temporal End"]+1, p["Temporal Step"])):
        # check if timestep already exists
        with h5py.File(h5py_path, "r") as h5py_file:
            time_step = (t-1) // p["Temporal Step"]
            data = h5py_file[f"sims/sim0/{time_step}"]
            if np.any(data): # ensure that the timestep is not empty, since h5py intializes with zeros
                h5py_file.close()
                print("Timestep %d already exists, skipping...\n" % t)
                continue
            h5py_file.close()

        # handler for worker processes
        slices = []
        def handle_result(result):
            slices.append(result)
        def handle_error(error):
            print("\n\nError occoured:")
            print(error)

        for tries in range(download_tries):
            # download 2d slices from 3d volume (in parallel) as the JHTDB API does not permit large 3d cutouts
            pool = mp.Pool(workers, maxtasksperchild=1)
            for x in range(start[0], end[0]+1, step[0]):
                if x in [s[0] for s in slices]:
                    continue

                slice_start = np.array([x, start[1], start[2]]).astype(np.int32)
                slice_end = np.array([x, end[1], end[2]]).astype(np.int32)
                slice_step = np.array([1, step[1], step[2]]).astype(np.int32)

                pool.apply_async(wrapped_download, args=(token, dataset["name"], field_code, t, slice_start, slice_end, slice_step), callback=handle_result, error_callback=handle_error)

            # check regularly if all slices have been downloaded
            waited_min = 0
            while len(slices) < len(range(start[0], end[0]+1, step[0])) and waited_min < timeout_min:
                time.sleep(poll_interval_min*60)
                waited_min += poll_interval_min

            # finish if all slices have been downloaded, otherwise retry with new pool
            pool.close()
            if len(slices) >= len(range(start[0], end[0]+1, step[0])):
                pool.join()
                break
            else:
                print("Missing slices, retrying for the %dth time..." % tries)
                pool.terminate()
                pool.join()
                if tries == download_tries-1:
                    raise TimeoutError("Failed to download all slices after %d tries, aborting..." % download_tries)

        channels = len(p["Fields"])
        res_x = 1 + ((p["Spatial End"][0] - p["Spatial Start"][0]) // p["Spatial Step"][0])
        res_y = 1 + ((p["Spatial End"][1] - p["Spatial Start"][1]) // p["Spatial Step"][1])
        res_z = 1 + ((p["Spatial End"][2] - p["Spatial Start"][2]) // p["Spatial Step"][2])

        # write timestep to h5py file
        with h5py.File(h5py_path, "a") as h5py_file:

            slices = sorted(slices, key=lambda x: x[0])
            data = np.concatenate([s[1] for s in slices], axis=2)
            data = np.transpose(data, (3,2,1,0)) # move channels to the front, jhtdb transposes x and z axes

            # data = np.zeros(shape=(channels, res_x, res_y, res_z))

            time_step = (t-1) // p["Temporal Step"]

            h5py_file[f"sims/sim0/{time_step}"][:] = data # np.expand_dims(data, axis=0)

            h5py_file.close()
            print("Timestep %d (index: %d) with shape %s written to disk\n" % (t, (t-1)//p["Temporal Step"], str(data.shape)))


    # reload data partially to render, as full dataset may be too large for RAM
    time_steps = (p["Temporal End"]+1 - p["Temporal Start"]) // p["Temporal Step"]
    steps_plot = min(10, time_steps)

    data = []
    with h5py.File(h5py_path, "r") as h5py_file:
        for t in range(0, time_steps, time_steps // steps_plot):
            data.append(h5py_file[f"sims/sim0/{t}"])
        h5py_file.close()

    if dataset_name == "mhd1024":
        vmin, vmax = -0.7, 0.7
    elif dataset_name == "isotropic1024coarse":
        vmin, vmax = -1.2, 1.2
    else:
        # automatically determined via min and max of data
        vmin, vmax = None, None

    render_trajectory(
        data=data,
        dimension=3,
        output_path=out_dir,
        sim_id=0,
        time_steps=len(data),
        steps_plot=len(data),
        vmin=vmin,
        vmax=vmax,
    )




if __name__ == '__main__':
    download_data(args.dataset_name, args.out_name, args.out_path, args.token)