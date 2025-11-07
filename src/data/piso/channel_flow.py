import numpy as np
from render import render_trajectory

DATA_DIR = '/mnt/data/pbdl-datasets-local/3d_channel_flow/'
SUBDIR = 'raw/'

DATASETS = [
            '400', '420', '440', '460', '480',
            '500', '520', '540', '560', '580',
            '600', '620', '640', '660', '680',
            '700', '720', '740', '760', '780',
            '800',
            '200', '300', '900', '1000',
            ]

dset = DATASETS[0]

import h5py
from tqdm import tqdm

def get_channel_flow():
    return {
        "PDE": "PISO: Channel Flow",
        "Dimension": 3,
        "Fields Scheme": "VVVpDDD",
        "Fields": ["Velocity X", "Velocity Y", "Velocity Z", "Pressure", "Deformation XX", "Deformation YY", "Deformation ZZ"],
        "Domain Extent": [2*np.pi, 2, np.pi],
        "Resolution": [192, 96, 96],
        "Time Steps": 200,
        "Dt": 0.1,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive", "z negative", "z positive"],
        "Boundary Conditions": ["open", "open", "open", "open", "open", "open"],
        "Constants": ["Reynolds Number"],
        "Reynolds Number (range)": [200, 1000],
    }

fields = []

import numpy as np

if __name__ == "__main__":

    with h5py.File(f'{DATA_DIR}channel_flow.hdf5', 'w') as file:

        for key in fields:
            file.create_dataset(key, data=None)

        metadata = get_channel_flow()

        group = file.create_group("sims", track_order=True)

        for key in metadata:
            group.attrs[key] = metadata[key]

        for sim, dataset_re in enumerate(DATASETS):

            print("Processing Re", dataset_re)

            data_list = []

            for i in tqdm(range(0, 200)):

                data = np.load(f'{DATA_DIR}/{SUBDIR}/channel-flow-Re{dataset_re}/domain_frame_{i:06d}.npz')

                velocity_data = data['1']
                pressure_data = data['2']

                dx = data['4'][0, 0, :-1] - data['4'][0, 0, 1:]
                dx = dx[:, :-1, :-1][None][None]

                dy = data['4'][0, 1, :, :-1] - data['4'][0, 1, :, 1:]
                dy = dy[:-1, :, :-1][None][None]

                dz = data['4'][0, 2, :, :, :-1] - data['4'][0, 2, :, :, 1:]
                dz = dz[:-1, :-1, :][None][None]

                data = np.concatenate((velocity_data, pressure_data, dx, dy, dz), axis=1)[0]
                data_list.append(data)

                # convert data to 16bit precision
                data = data.astype('float16')

                data = np.transpose(data, (1, 2, 3, 0))

                num_channels = data.shape[-1]

                dataset = file.create_dataset(f"sims/sim{sim}/{i}", data=data, dtype='float16',
                                                  chunks=(64, 64, 64, num_channels))

                dataset.attrs['Reynolds Number'] = int(dataset_re)

            vmin = 0
            vmax = 1

            data = np.stack(data_list, axis=0)
            render_trajectory(
                data=data[:, :4],
                dimension=3,
                output_path=DATA_DIR + 'rendered/',
                sim_id=int(dataset_re),
                time_steps=20,
                steps_plot=6,
                vmin=vmin,
                vmax=vmax,
            )

        file.close()