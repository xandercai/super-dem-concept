from sklearn.metrics import jaccard_score
import torch


def oa(pred, y):
    flat_y = y.squeeze()
    flat_pred = pred.argmax(dim=1)
    acc = torch.count_nonzero(flat_y == flat_pred) / torch.numel(flat_y)
    return acc.cpu().numpy()


def iou(pred, y):
    flat_y = y.cpu().numpy().squeeze()
    flat_pred = pred.argmax(dim=1).detach().cpu().numpy()
    return jaccard_score(
        flat_y.reshape(-1), flat_pred.reshape(-1), zero_division=1.0, average="macro"
    )


def loss(p, t):
    ce = torch.nn.CrossEntropyLoss()
    return ce(p, t.squeeze().type(torch.long))


if __name__ == "__main__":
    n_class = 10
    preds = torch.randn(4, n_class, 24, 24)
    labels = torch.empty(4, 24, 24, dtype=torch.long).random_(n_class)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(preds, labels)
    print(loss)
