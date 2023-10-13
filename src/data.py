from pathlib import Path
import rasterio as rio
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, Units
from typing import List
import torch
import json
from pyproj import CRS
from easydict import EasyDict as edict
from typing import Union
import numpy as np
from matplotlib import pyplot as plt

global_crs = CRS.from_epsg(32759)


def scale_image(item: dict):
    item["image"] = item["image"] / 10_000
    return item


def calc_statistics(
    dset: RasterDataset,
    is_image: bool = False,
    has_nan: bool = False,
    unique: bool = False,
):
    """
    Calculate the statistics (mean and std) for the entire dataset
    Warning: This is an approximation. The correct value should take into account the
    mean for the whole dataset for computing individual stds.
    For correctness I suggest checking: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    """

    # To avoid loading the entire dataset in memory, we will loop through each img
    # The filenames will be retrieved from the dataset's rtree index
    files = [
        item.object for item in dset.index.intersection(dset.index.bounds, objects=True)
    ]

    # Reseting statistics
    accum_mean = 0
    accum_std = 0

    for file in files:
        if is_image:
            arr = rio.open(file).read() / 10_000
        else:
            arr = rio.open(file).read()
        if has_nan:
            arr = np.nan_to_num(arr, nan=0)
        if unique:
            print(file)
            print(np.unique(arr))
            print(np.where(arr == 0))
        accum_mean += arr.reshape((arr.shape[0], -1)).mean(axis=1)
        accum_std += arr.reshape((arr.shape[0], -1)).std(axis=1)

    # at the end, we shall have 2 vectors with lenght n=chnls
    # we will average them considering the number of images
    return accum_mean / len(files), accum_std / len(files)


class Normalizer(torch.nn.Module):
    def __init__(self, mean: List[float], stdev: List[float]):
        super().__init__()
        self.mean = torch.Tensor(mean)[:, None, None]
        self.std = torch.Tensor(stdev)[:, None, None]

    def forward(self, inputs: torch.Tensor):
        """
        Normalize the batch.
        """
        x = inputs[..., : len(self.mean), :, :]
        # if batch
        if inputs.ndim == 4:
            x = (x - self.mean[None, ...]) / self.std[None, ...]
        else:
            x = (x - self.mean) / self.std
        inputs[..., : len(self.mean), :, :] = x
        return inputs

    def revert(self, inputs: torch.Tensor):
        """
        De-normalize the batch.
        """
        x = inputs[..., : len(self.mean), :, :]
        # if batch
        if x.ndim == 4:
            x = inputs[:, : len(self.mean), ...]
            x = x * self.std[None, ...] + self.mean[None, ...]
        else:
            x = x * self.std + self.mean
        inputs[..., : len(self.mean), :, :] = x
        return inputs


class NanCleaner(torch.nn.Module):
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose

    def forward(self, inputs: torch.Tensor):
        """
        Clean Nan if any
        """
        shape = inputs.shape
        batch_size = inputs.shape[0]
        channel_size = inputs.shape[1]
        spatial_size = inputs.shape[2:]
        inputs = inputs.view(batch_size, channel_size, -1)
        nan_arr = torch.any(torch.isnan(inputs), dim=2)
        if nan_arr.any():
            if self.verbose:
                nan_idx = np.where(nan_arr.cpu().numpy())
                for i in range(len(nan_idx[0])):
                    print(
                        f"\t\tGot nan at: batch {nan_idx[0][i]}, channel {nan_idx[1][i]}"
                    )
                    nan_im = (
                        inputs[nan_idx[0][i], nan_idx[1][i], :]
                        .view(*spatial_size)
                        .cpu()
                        .numpy()
                        .T
                    )
                    plt.imshow(nan_im, cmap="terrain")
                    plt.show()
            inputs = torch.nan_to_num(inputs)
        inputs = inputs.view(shape)
        return inputs


def get_paths(root_dir: Union[Path, str]):
    """
    Get the paths of the datasets and subsets.
    """
    dataset_list = [d for d in Path(root_dir).glob("*") if d.is_dir()]
    dataset_name = [d.name for d in dataset_list]
    subset_dict_list = []
    for d in dataset_list:
        subset_path_list = [s.as_posix() for s in d.glob("*") if s.is_dir()]
        subset_name_list = [Path(s).name for s in subset_path_list]
        subset_dict_list.append(dict(zip(subset_name_list, subset_path_list)))
    return edict(dict(zip(dataset_name, subset_dict_list)))


class Planet(RasterDataset):
    filename_glob = "*3B_AnalyticMS_SR_8b*.tif"
    is_image = True
    separate_files = False
    all_bands = [
        "coastal_blue",
        "blue",
        "green_i",
        "green",
        "yellow",
        "red",
        "rededge",
        "nir",
    ]
    rgb_bands = ["red", "green", "blue"]


def get_dataset(split: str, p: edict):
    """
    generate dataset
    """
    if isinstance(split, str):
        split = [split]
    else:
        split = sorted(split)

    if set(split).issubset({"train", "val", "test"}):
        # dataset directories
        dataset_path_dict = get_paths(p.dataset_dir)

        if split == ["train"]:
            dataset_name_list = [
                k for k in dataset_path_dict.keys() if k in p.train_set_name
            ]
            print("train", dataset_name_list)
        else:
            dataset_name_list = [
                k for k in dataset_path_dict.keys() if k in p.test_set_name
            ]
            print("test", dataset_name_list)
    else:
        raise ValueError(
            f"Invalid split name: {split} (must be 'train', 'val' or 'test')."
        )

    dataset_list = []
    if Path(p.dataset_dir).name.lower() == "dem_nz":
        for dataset_name in dataset_name_list:
            if len(p.tasks) == 1 and p.tasks[0] == "semseg":
                image_set = Planet(
                    paths=dataset_path_dict[dataset_name].image,
                    crs=global_crs,
                    res=p.resolution,
                    bands=["red", "green", "blue"] if p.rgb_only else None,
                    transforms=scale_image,
                )
                dem_lr_set = RasterDataset(
                    paths=dataset_path_dict[dataset_name].dem_lr,
                    crs=global_crs,
                    res=p.resolution,
                )
                canopy_set = RasterDataset(
                    paths=dataset_path_dict[dataset_name].canopy,
                    crs=global_crs,
                    res=p.resolution,
                )
                building_set = RasterDataset(
                    paths=dataset_path_dict[dataset_name].building,
                    crs=global_crs,
                    res=p.resolution,
                )
                # dem_hr_set = RasterDataset(
                #     paths=dataset_path_dict[dataset_name].dem_hr,
                #     crs=dem_hr_crs,
                #     res=p.resolution,
                # )
                mask_set = RasterDataset(
                    paths=dataset_path_dict[dataset_name].mask,
                    crs=global_crs,
                    res=p.resolution,
                )

                mask_set.is_image = False
                # dem_hr_set.is_image = False

                dataset = (
                    image_set
                    & dem_lr_set
                    & canopy_set
                    & building_set
                    # & dem_hr_set
                    & mask_set
                )
            dataset_list.append(dataset)

        if len(dataset_list) > 1:
            dataset = dataset_list[0] | dataset_list[1]
        else:
            dataset = dataset_list[0]

    # Display stats
    print(f"{split} dataset size: {len(dataset)}.")

    return dataset


def get_dataloader(dataset: RasterDataset, p: edict):
    """
    generate dataloader
    """
    sample_scale = p.sample_scale
    sample_size = p.sample_size

    sampler = RandomGeoSampler(
        dataset,
        size=sample_size,
        length=len(dataset) * sample_scale,
        units=Units.PIXELS,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=p.batch_size,
        collate_fn=stack_samples,
    )
    print(f"dataloader length: {len(dataset) * sample_scale}.")
    return dataloader


def get_statistics(root, p):
    image_set = Planet(
        paths=(Path(root) / Path("image")).as_posix(),
        crs=global_crs,
        res=p.resolution,
    )
    # [0.02125753 0.0357015  0.04440632 0.05297579 0.05467798 0.05113702 0.09292307 0.24820472] [0.01749222 0.02163442 0.02477142 0.02770515 0.0315022  0.03341371 0.04396875 0.11720941]
    mean, std = calc_statistics(image_set, is_image=True)
    print("image stat", mean, std)

    dem_lr_set = RasterDataset(
        paths=(Path(root) / Path("dem_lr")).as_posix(),
        crs=global_crs,
        res=p.resolution,
    )
    # [420.23532] [267.00613]
    mean, std = calc_statistics(dem_lr_set, has_nan=True)
    print("cop30 stat", mean, std)

    canopy_set = RasterDataset(
        paths=(Path(root) / Path("canopy")).as_posix(),
        crs=global_crs,
        res=p.resolution,
    )
    # [4.96576484] [5.874813]
    mean, std = calc_statistics(canopy_set)
    print("canopy stat", mean, std)

    building_set = RasterDataset(
        paths=(Path(root) / Path("building")).as_posix(),
        crs=global_crs,
        res=p.resolution,
    )
    # [1.00909591] [0.05473682]
    mean, std = calc_statistics(building_set)
    print("building stat", mean, std)

    mask_set = RasterDataset(
        paths=(Path(root) / Path("mask")).as_posix(),
        crs=global_crs,
        res=p.resolution,
    )
    # [6.4514706] [3.22235]
    mean, std = calc_statistics(mask_set, unique=True)
    print("building stat", mean, std)


if __name__ == "__main__":
    import utils
    from src.plot import plot_batch

    p = utils.create_config("./config/dem_nz/deeplabv3_resnet50/semseg.yml")
    # dataset = get_dataset("train", p)
    # dataloader = get_dataloader(dataset, p)
    # batch = next(iter(dataloader))
    # print("verify a batch", batch.keys(), batch["image"].shape, batch["mask"].shape)
    # plot_batch(batch.copy(), bright=5, chnls=[0, 1, 2])
    root = r"../datasets/dem_nz_stat"
    get_statistics(root, p)
