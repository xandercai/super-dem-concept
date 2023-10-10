import torch


def get_optimizer(model, p):
    """
    get optimizer
    """
    optimizer = None

    if p.optimizer == "Adam":
        pass

    if p.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=p.lr, weight_decay=5e-4, momentum=0.9
        )

    return optimizer


def get_scheduler(optimizer, dataset, p):
    """
    get scheduler
    """
    scheduler = None

    if p.scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=p.lr,
            steps_per_epoch=len(dataset) // p.batch_size,
            epochs=p.epochs,
            anneal_strategy="linear",
        )

    if p.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=p.epochs, eta_min=0
        )

    if p.scheduler == "ConstantLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)

    return scheduler
