from datetime import datetime
import numpy as np
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_loop(
    epochs: int,
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    acc_fns: Optional[List] = None,
    batch_tfms: Optional[Callable] = None,
    device: Optional[str] = "cuda",
    sample_scale: int = 1,
    verbose: bool = False,
) -> List[float]:
    acc = []
    model.to(device)
    print(
        f"Epoch: {0:02d}\tLR: {scheduler.get_last_lr()[0]:7.5f}\tLoss: {0:7.5f}\t"
        f"Accs={[f'{a:5.3f}' for a in [0.0] * len(acc_fns)]}\tTime: 0"
    )
    for epoch in range(epochs):
        epoch_start = datetime.now()
        accum_loss = 0
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        for i, batch in enumerate(train_dl):
            step_start = datetime.now()
            if batch_tfms is not None:
                batch["image"] = batch_tfms(batch["image"])
            X = batch["image"].to(device)
            y = batch["mask"].type(torch.long).to(device)

            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            pred = model(X)["out"]
            loss = loss_fn(pred, y)

            # BackProp
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # update the accum loss
            step_loss = float(loss / len(batch))
            accum_loss += step_loss
            if verbose:
                print(
                    f"\tEpoch: {epoch+1:02d}\tStep: {i+1:02d}\tLR: {lr:7.5f}\tLoss: {step_loss:7.5f}\t"
                    f"Time: {datetime.now() - step_start}"
                )

        # Testing against the validation dataset
        if acc_fns is not None and val_dl is not None:
            # reset the accuracies metrics
            acc = [0.0] * len(acc_fns)

            with torch.no_grad():
                model.eval()
                for batch in val_dl:
                    if batch_tfms is not None:
                        batch["image"] = batch_tfms(batch["image"])

                    X = batch["image"].type(torch.float32).to(device)
                    y = batch["mask"].type(torch.long).to(device)

                    pred = model(X)["out"]

                    for i, acc_fn in enumerate(acc_fns):
                        acc[i] = acc[i] + float(
                            acc_fn(pred, y) / (len(val_dl.dataset) * sample_scale)
                        )

        # at the end of the epoch, print the loss, etc.
        print(
            f"Epoch: {epoch+1:02d}\tLR: {lr:7.5f}\tLoss: {accum_loss:7.5f}\tAccs={[f'{a:5.3f}' for a in acc]}\t"
            f"Time: {(datetime.now() - epoch_start)}"
        )

    return acc
