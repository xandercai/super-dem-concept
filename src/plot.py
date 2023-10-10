from typing import Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import torch
from torchgeo.datasets import unbind_samples
import numpy as np
from easydict import EasyDict as edict


def plot_images(
    images: Iterable, axs: Iterable, chnls: List[int] = [2, 1, 0], bright: float = 4.0
):
    for img, ax in zip(images, axs):
        arr = torch.clamp(bright * img, min=0, max=1).numpy()
        rgb = arr.transpose(1, 2, 0)[:, :, chnls]
        ax.imshow(rgb)
        ax.axis("off")


def plot_masks(masks: Iterable, axs: Iterable, cmap, nodata=np.nan):
    for mask, ax in zip(masks, axs):
        arr = mask.squeeze().numpy()
        arr[arr == nodata] = 0
        ax.imshow(arr, cmap=cmap)
        ax.axis("off")


def plot_batch(
    batch: Union[dict, torch.Tensor],
    p: edict,
    masks: Optional[torch.Tensor] = None,
    bright: float = 5.0,
    cols: int = 6,
    width: int = 5,
    chnls: List[int] = [0, 1, 2],
    limit: int = 4,
):
    batch = batch.copy()
    if isinstance(batch, dict):
        # Get the samples and the number of items in the batch
        # samples = unbind_samples(batch)
        samples = unbind_samples(batch)[:limit]
        # if batch contains images and masks, the number of images will be doubled
        n = (
            6 * len(samples)
            if ("image" in batch) and ("mask" in batch)
            else 4 * len(samples)
        )
        # calculate the number of rows in the grid
        img_samples = (
            map(lambda x: x["image"][:3], samples) if "image" in batch else None
        )
        del_samples = (
            map(lambda x: x["image"][3], samples) if "image" in batch else None
        )
        can_samples = (
            map(lambda x: x["image"][4], samples) if "image" in batch else None
        )
        bui_samples = (
            map(lambda x: x["image"][5], samples) if "image" in batch else None
        )
        if len(p.tasks) > 1 or "semseg" not in p.tasks:
            deh_samples = (
                map(lambda x: x["mask"][0], samples) if "image" in batch else None
            )
        if "semseg" in p.tasks:
            msk_samples = (
                map(lambda x: x["mask"][0], samples) if "mask" in batch else None
            )
    else:
        n = 2 * len(batch) if masks is not None else len(batch)
        img_samples = batch
        msk_samples = masks if masks is not None else None
    rows = n // cols + (1 if n % cols != 0 else 0)
    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if img_samples is not None and msk_samples is not None:
        plot_images(
            images=img_samples,
            axs=axs.reshape(-1)[::6],
            chnls=chnls,
            bright=bright,
        )
        plot_masks(masks=del_samples, axs=axs.reshape(-1)[1::6], cmap="terrain")
        plot_masks(
            masks=can_samples, axs=axs.reshape(-1)[2::6], cmap="Greens", nodata=101
        )
        plot_masks(masks=bui_samples, axs=axs.reshape(-1)[3::6], cmap="tab10", nodata=0)
        if len(p.tasks) > 1 or "semseg" not in p.tasks:
            plot_masks(masks=deh_samples, axs=axs.reshape(-1)[4::6], cmap="terrain")
        if "semseg" in p.tasks:
            plot_masks(
                masks=msk_samples, axs=axs.reshape(-1)[4::6], cmap="Paired", nodata=0
            )
    else:
        if img_samples is not None:
            plot_images(
                images=img_samples,
                axs=axs.reshape(-1),
                chnls=chnls,
                bright=bright,
            )
        elif msk_samples is not None:
            plot_masks(masks=msk_samples, axs=axs.reshape(-1))
    plt.show()
