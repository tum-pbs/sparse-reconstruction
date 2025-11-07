from typing import Optional

import torch
from torch.utils.data import random_split
from torchvision.transforms.v2 import Transform, ToDtype, Compose, Lambda, Normalize, RandomHorizontalFlip, RandomVerticalFlip
from pbdl.torch.dataset import Dataset as PBDLDataset
from pbdl.torch.dataset import Dataset3D as PBDLDataset3D

from .variable_dt_dataset import VariableDtDataset


'''
    3D Datasets downloaded from the JHTDB
'''
seed = 45


def jhtdb_3d_datasets(dataset_name: str,
                 dataset_directory: str,
                 unrolling_steps: int,
                 intermediate_time_steps: bool = True,
                 variable_dt_stride_maximum: int = 1,
                 test_variable_dt_stride_maximum: int = 1,
                 test_unrolling_steps: Optional[int] = None,
                 test_intermediate_time_steps: Optional[bool] = None,
                 normalize_data: Optional[str] = None,
                 normalize_const: Optional[str] = None,
                 crop_size: Optional[int] = None) -> tuple[PBDLDataset, PBDLDataset, PBDLDataset]:
    r'''
    Creates JHTDB train, val, and test dataset objects.

    Args:
        dataset_name: name of local dataset file
        dataset_directory: directory where the data set is located
        unrolling_steps: number of time steps between start and end of sequence
        intermediate_time_steps: determines if intermediate time steps are included in the data or not
        variable_dt_stride_maximum: maximum number of time steps between start and end of sequence for variable dt training
        test_variable_dt_stride_maximum: maximum number of time steps between start and end of sequence for variable dt testing
        test_unrolling_steps: number of time steps between start and end of sequence for the test dataset
        test_intermediate_time_steps: determines if intermediate time steps are included in the data
                                        or not for the test dataset
        normalize_data: type of normalization to apply to the data (mean-std, std, zero-to-one, minus-one-to-one, None)
        normalize_const: type of normalization to apply to the constants (mean-std, std, zero-to-one, minus-one-to-one, None)
        crop_size: size of the cropped region around the center of the domain. If None, no cropping is applied.

    Returns:
        tuple[PBDLDataset, PBDLDataset, PBDLDataset]: train, validation and test datasets
    '''

    if test_unrolling_steps is None:
        test_unrolling_steps = unrolling_steps
    if test_intermediate_time_steps is None:
        test_intermediate_time_steps = intermediate_time_steps
    if test_variable_dt_stride_maximum is None:
        test_variable_dt_stride_maximum = variable_dt_stride_maximum

    # single simulation, use different temporal intervals for training and testing
    if dataset_name == "channel":
        train_trim = 60
        test_trim = 400 - train_trim
    elif dataset_name == "isotropic1024coarse":
        train_trim = 80
        test_trim = 500 - train_trim
    elif dataset_name == "mhd1024":
        train_trim = 20
        test_trim = 100 - train_trim
    elif dataset_name == "transition_bl":
        train_trim = 150
        test_trim = 940 - train_trim
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    params_train = {
        "dset_name": dataset_name,
        "local_datasets_dir": dataset_directory,
        "trim_end": train_trim,
        "time_steps": unrolling_steps,
        "intermediate_time_steps": intermediate_time_steps,
        "normalize_const": normalize_const,
        "normalize_data": normalize_data,
        "crop_size": crop_size,
    }

    if variable_dt_stride_maximum <= 1:
        pbdl_all = PBDLDataset3D(**params_train)
    else:
        pbdl_all = VariableDtDataset(**params_train, maximum_dt=variable_dt_stride_maximum, seed=None)

    train, val = random_split(
        pbdl_all, [0.85, 0.15], generator=torch.Generator().manual_seed(seed)
    )
    # the data indices have to be sorted manually since random_split shuffles them
    train.indices = sorted(train.indices)
    val.indices = sorted(val.indices)

    params_test = {
        "dset_name": dataset_name,
        "local_datasets_dir": dataset_directory,
        "trim_start": test_trim,
        "time_steps": test_unrolling_steps,
        "intermediate_time_steps": test_intermediate_time_steps,
        "normalize_const": normalize_const,
        "normalize_data": normalize_data,
        "crop_size": crop_size,
    }

    if test_variable_dt_stride_maximum <= 1:
        test = PBDLDataset3D(**params_test)
    else:
        test = VariableDtDataset(**params_test, maximum_dt=test_variable_dt_stride_maximum, seed=seed)

    return train, val, test



def jhtdb_3d_transforms(dataset_name: str)-> tuple[Transform, Transform, Transform]:
    r'''
    Creates JHTDB train, val, and test transform objects.

    Args:
        dataset_name: name of local dataset

    Returns:
        tuple[Transform, Transform, Transform]: train, validation and test transforms
    '''

    transform_train = Compose(
        [
            ToDtype(torch.float32),
        ]
    )

    return transform_train, transform_train, transform_train


