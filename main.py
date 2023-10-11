from pathlib import Path

import torch
from osgeo import gdal
from torchgeo.transforms import indices

from src.data import get_dataset, get_dataloader, Normalizer, NanCleaner
from src.loss import loss, oa, iou
from src.model import get_model
from src.plot import plot_batch
from src.train import train_loop
from src.utils import create_config
from src.optim import get_scheduler, get_optimizer
from utils import Logger
import sys
import os
import argparse
import json

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

gdal.PushErrorHandler(
    "CPLQuietErrorHandler"
)  # for suppressing "TIFFReadDirectory..." warning

# Parser
parser = argparse.ArgumentParser(description="Vanilla Training")
parser.add_argument("--config_file", help="Config file for running the experiment")
args = parser.parse_args()


def main():
    # Part 0. Create config
    p = create_config(args.config_file)
    sys.stdout = Logger(os.path.join(p.result_dir, "log_file.txt"))

    # Part 1. Build dataloader
    train_dataset = get_dataset("train", p)
    train_dataloader = get_dataloader(train_dataset, p)

    test_dataset = get_dataset("test", p)
    test_dataloader = get_dataloader(test_dataset, p)

    # Part 2. Build transformers
    # transformers is defined in src/data.py
    normalizer = Normalizer(
        mean=p.mean,
        stdev=p.std,
    )
    nan_cleaner = NanCleaner(
        verbose=p.verbose,
    )

    # verify normalizer
    train_batch = next(iter(train_dataloader))
    veri_batch = train_batch.copy()
    veri_batch["image"] = normalizer(veri_batch["image"])
    print(
        "verify a normed veri_batch",
        veri_batch.keys(),
        veri_batch["image"].shape,
        veri_batch["mask"].shape,
    )

    if p.verbose:
        plot_batch(veri_batch, p)
    revert_batch = veri_batch.copy()
    revert_batch["image"] = normalizer.revert(revert_batch["image"])
    print(
        "verify a revert veri_batch",
        revert_batch.keys(),
        revert_batch["image"].shape,
        revert_batch["mask"].shape,
    )
    if p.verbose:
        plot_batch(revert_batch, p)

    transformers = torch.nn.Sequential(
        # indices.AppendNDWI(index_green=1, index_nir=3),
        # indices.AppendNDWI(index_green=1, index_nir=5),
        # indices.AppendNDVI(index_nir=3, index_red=2),
        normalizer,
        nan_cleaner,
    )

    # Part 3. Create model
    # model is defined in src/model.py
    model = get_model(p.num_classes, p.model_name, p.tasks, init_random=p.init_random)
    print("Model name:", model.name)
    if p.verbose:
        print(model)

    # Part 4. Loss, metrics, and optimizer
    # loss is defined in src/loss.py, cross entropy
    # metrics are overall-accuracy and intersection-over-union, oa and iou are defined in src/loss.py
    optimizer = get_optimizer(model, p)
    scheduler = get_scheduler(optimizer, train_dataloader, p)

    # Part 5. Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)

    acc = train_loop(
        epochs=p.epochs,
        train_dl=train_dataloader,
        val_dl=test_dataloader,
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        acc_fns=[oa, iou],
        batch_tfms=transformers,
        device=device,
        sample_scale=p.sample_scale,
        parallel=p.parallel,
        verbose=p.verbose,
    )

    # Part 6. Save weights
    file_name = "acc_" + "_".join([str(i)[:5] for i in acc]) + ".pth"
    file_path = Path(p.result_dir) / Path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), file_path)


if __name__ == "__main__":
    main()
