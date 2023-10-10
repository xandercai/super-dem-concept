import torch.nn.init as init
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50
import torch


def init_weights_random(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
    return model


def get_model(num_classes: int, model_name, tasks, init_random: bool = False):
    Model = None
    if model_name == "deeplabv3_resnet50" and len(tasks) == 1 and tasks[0] == "semseg":
        Model = deeplabv3_resnet50(weights=None, num_classes=num_classes)

        backbone = Model.get_submodule("backbone")
        conv = nn.modules.conv.Conv2d(
            in_channels=6,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        backbone.register_module("conv1", conv)

    Model.name = model_name

    if init_random:
        Model = init_weights_random(Model)

    return Model


if __name__ == "__main__":
    random_data = torch.randn(4, *(6, 256, 256))
    model = get_model(11)
    print(model)
    out = model(random_data)
    print(out.keys())
    print(out["out"].shape)
