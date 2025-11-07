import warnings

import lightning.pytorch
from torchvision.transforms.v2 import ToDtype, Lambda, Compose

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pl_bolts.datamodules import CIFAR10DataModule
    from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

import torchvision
import torch

from torchvision.transforms import Normalize
import torchvision.transforms.functional as TF

# this seems a bit unnecessary, but in fact CIFAR10DataModule is a subclass of pytorch_lightning.LightningDataModule,
# but we need a subclass of lightning.pytorch.LightningDataModule
class CIFAR10DataModuleWrapper(CIFAR10DataModule, lightning.pytorch.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

from .utils import UnNormalize

def get_CIFAR10_post_transform(normalize: bool = True):

    if normalize:
        norm_ = UnNormalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )
    else:
        norm_ = UnNormalize(
            mean=[x / 255.0 for x in [127.5, 127.5, 127.5]],
            std=[x / 255.0 for x in [127.5, 127.5, 127.5]]
        )

    return Compose(
        [
            norm_,
            Lambda(lambda x: torch.clamp(255 * x, 0, 255)),
            ToDtype(torch.uint8, scale=False)
        ]
    )

image_scale = Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

def prepare_plots(normalize: bool = True):
    fn = get_CIFAR10_post_transform(normalize=normalize)
    return fn

def get_CIFAR10(
        directory: str,
        batch_size: int,
        num_workers: int = 2,
        normalize: bool = True,
        val_split: float = 0.0
    ) -> CIFAR10DataModule:
    """
    Get CIFAR10 data module
    :param directory: Location to store data
    :param batch_size: Batch size
    :param num_workers: Number of workers
    :param normalize: Whether to normalize the data to a normal Gaussian or just rescale it to [-1, 1]
    :param val_split: Validation split
    :return: CIFAR10 data module
    """

    train_transforms_ = [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]

    if normalize:
        train_transforms_.append(cifar10_normalization())
    else:
        train_transforms_.append(image_scale)

    train_transforms = torchvision.transforms.Compose(
        train_transforms_
    )

    test_transforms_ = [
        torchvision.transforms.ToTensor(),
    ]

    if normalize:
        test_transforms_.append(cifar10_normalization())
    else:
        test_transforms_.append(image_scale)

    test_transforms = torchvision.transforms.Compose(
        test_transforms_
    )

    module = CIFAR10DataModuleWrapper(
        data_dir=directory,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    module.EXTRA_ARGS["download"] = True

    return module

class RotationTransform:

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

def get_CIFAR10_rotated(
        directory: str,
        batch_size: int,
        num_workers: int = 2,
        normalize: bool = True,
        val_split: float = 0.0
    ) -> CIFAR10DataModule:
    """
    Get CIFAR10 data module
    :param directory: Location to store data
    :param batch_size: Batch size
    :param num_workers: Number of workers
    :param normalize: Whether to normalize the data to a normal Gaussian or just rescale it to [-1, 1]
    :param val_split: Validation split
    :return: CIFAR10 data module
    """

    train_transforms_ = [
        torchvision.transforms.RandomHorizontalFlip(),
        RotationTransform(90),
        torchvision.transforms.ToTensor(),
    ]

    if normalize:
        train_transforms_.append(cifar10_normalization())
    else:
        train_transforms_.append(image_scale)

    train_transforms = torchvision.transforms.Compose(
        train_transforms_
    )

    test_transforms_ = [
        RotationTransform(90),
        torchvision.transforms.ToTensor(),
    ]

    if normalize:
        test_transforms_.append(cifar10_normalization())
    else:
        test_transforms_.append(image_scale)

    test_transforms = torchvision.transforms.Compose(
        test_transforms_
    )

    module = CIFAR10DataModuleWrapper(
        data_dir=directory,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )

    module.EXTRA_ARGS["download"] = True

    return module