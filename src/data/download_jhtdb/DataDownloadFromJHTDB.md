# Installation
The [pyJHTDB package](https://github.com/idies/pyJHTDB) requires old versions of python and numpy, so it is highly recommended to create a separate conda environment for downloading JHTDB data. To install pyJHTDB follow these steps:
```bash
conda create -n jhtdb python=3.6
conda activate jhtdb
```
Install required packages:
```bash
pip install -r src/data/download_jhtdb/requirementsJHTDB.txt
pip install pyJHTDB
pip install vape4d
```
[vape4d](https://pypi.org/project/vape4d/) is not strictly required and only used to render the volumes as a final step.

# Usage
Downloading of volumetric turbulence datasets is performed via the script `download_jhtdb_3d.py`. 2D slices of the 3D data can be downloaded with `download_jhtdb_2d.py`. Both scripts accept the following arguments:

- `--dataset_name`: Specifies the dataset to download.
- `--out_name`: Specifies the name of the output dataset file.
- `--out_path`: Specifies the directory where the output files will be saved.
- `--token`: The JHTDB access token.

Some additional options regarding the download details (timeouts, tries, worker thread, etc.) can be found inside the `download_data` functions. Please consider looking at the detailed effects in the remaining code before changing these values.

`render.py` contains functionality to render the generated data. Running `download_jhtdb.py` automatically creates a single image with different visualizations for downloaded simulation of the dataset. The rendered images are saved in a directory with name `--out_name` to the location specified by `--out_path`. It is possible to manually render separate areas from the dataset (even while the download is still running) by running `render_existing.py` with suitable start, end, and step values.

`download_setups_2d.py` and `download_setups_3d.py` contain the basic parameters for downloads, which are imported by the download scripts. They also serve as an overview over supported datasets and parameter values. The name of the datasets in these file are the available values for `--dataset_name` in  `download_jhtdb_2d.py` and `download_jhtdb_3d.py`. The corresponding function for each PDE returns a JHTDB dataset info dictionary, fixed simulation and cutout parameters, and a field code that instructs the JHTDB servers on which fields to download for each simulation. Expanding or altering the simulation setups is straightforward by extending or modifying these files. Note that the choices for the cutout ranges for each PDE affects the spatial and temporal resolution of the output data.

`generate_all_2d.sh` and `generate_all_3d.sh` are bash script that can be used to download all datasets sequentially via calling the python scripts described above with the respective arguments.


# All datasets in the JHTDB with their corresponding maximum size [simulations s, time steps t, channels c, size x, size y, size z]
- Channel Flow (`channel`)
    - Data: [s=1, t=4000, c=4, x=2048, y=512, z=1536]
    - Channels: Velocity X/Y/Z, Pressure

- [CURRENTLY UNUSED!] Channel Flow at Reynolds Number 5200 (`channel5200`)
    - Data: [s=1, t=10, c=4, x=10240, y=1536, z=7680]
    - Channels: Velocity X/Y/Z, Pressure

- Forced Isotropic Turbulence (`isotropic1024coarse`)
    - Data: [s=1, t=5028, c=4, x=1024, y=1024, z=1024]
    - Channels: Velocity X/Y/Z, Pressure

- [CURRENTLY UNUSED!] Forced Isotropic Turbulence with fine temporal resolution (`isotropic1024fine`)
    - Data: [s=1, t=100, c=4, x=1024, y=1024, z=1024]
    - Channels: Velocity X/Y/Z, Pressure

- [CURRENTLY UNUSED!] Forced Isotropic Turbulence with high spatial resolution (`isotropic4096`)
    - Data: [s=1, t=1, c=3, x=4096, y=4096, z=4096]
    - Channels: Velocity X/Y/Z

- Forced Magnetohydrodynamic Turbulence (`mhd1024`)
    - Data: [s=1, t=1024, c=10, x=1024, y=1024, z=1024]
    - Channels: Velocity X/Y/Z, Pressure, Magnetic Field X/Y/Z, Vector Potential X/Y/Z

- [CURRENTLY UNUSED!] Rotating Stratified Turbulence (`rotstrat4096`)
    - Data: [s=1, t=5, c=4, x=4096, y=4096, z=4096]
    - Channels: Velocity X/Y/Z, Temperature

- Transitional Boundary Layer (`transition_bl`)
    - Data: [s=1, t=4701, c=4, x=10240, y=1536, z=2048] NOTE: only the [0:3320, 0:224, 0:2048] slice actually contains useable data
    - Channels: Velocity X/Y/Z, Pressure


# Downloaded JHTDB datasets in 2D
The following PDEs are supported in 2D at the moment. Each 2D simulation corresponds to a different z-slice of the 3D simulation.

- Channel Flow (`channel`)
    - Data: [s=5, t=400, c=4, x=2048, y=512]
    - Channels: Velocity X/Y/Z, Pressure

- Isotropic Turbulence (`isocoarse1024`)
    - Data: [s=5, t=500, c=4, x=1024, y=1024]
    - Channels: Velocity X/Y/Z, Pressure

- Magnetohydrodynamic Turbulence (`mhd1024`)
    - Data: [s=5, t=100, c=10, x=1024, y=1024]
    - Channels: Velocity X/Y/Z, Pressure, Magnetic Field X/Y/Z, Vector Potential X/Y/Z

- Transitional Boundary Layer (`transition_bl`)
    - Data: [s=5, t=940, c=4, x=2048, y=224]
    - Channels: Velocity X/Y/Z, Pressure


# Downloaded JHTDB datasets in 3D
The following PDEs are supported in 3D at the moment:

- Channel Flow (`channel`)
    - Data: [s=1, t=400, c=4, x=512, y=512, z=512]
    - Channels: Velocity X/Y/Z, Pressure

- Isotropic Turbulence (`isocoarse1024`)
    - Data: [s=1, t=500, c=4, x=512, y=512, z=512]
    - Channels: Velocity X/Y/Z, Pressure

- Magnetohydrodynamic Turbulence (`mhd1024`)
    - Data: [s=1, t=100, c=10, x=512, y=512, z=512]
    - Channels: Velocity X/Y/Z, Pressure, Magnetic Field X/Y/Z, Vector Potential X/Y/Z

- Transitional Boundary Layer (`transition_bl`)
    - Data: [s=1, t=940, c=4, x=224, y=224, z=224]
    - Channels: Velocity X/Y/Z, Pressure

The individual parameters for each downloaded dataset are described in more detail in `download_setups_2d.py` and `download_setups_3d.py`.