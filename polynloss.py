import torch

__all__ = [
    "polynloss",
    "polynfocal"
]


def crossentropyfirstn(probs, tgts, n=100, reduction='none'):
    losses = list()
    for prob, tgt in zip(probs, tgts):
        loss = 0.0
        for i in range(n):
            loss += 1/(i+1) * pow(1-prob, i+1)
        losses.append(loss[tgt].item())

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()

    return losses


def focallossfirstn(probs, tgts, gamma, n=100, reduction='none'):
    losses = list()
    for prob, tgt in zip(probs, tgts):
        loss = 0.0
        for i in range(n):
            loss += 1/(i+1) * pow(1-prob, gamma+i+1)
        losses.append(loss[tgt].item())

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()

    return losses


def polynloss(probs, tgts, eps, n=100, reduction='none'):
    drop_poly = crossentropyfirstn(probs, tgts, n, 'none')
    eps_losses = list()
    for prob, tgt in zip(probs, tgts):
        loss = 0.0
        for i in range(len(eps)):
            loss += eps[i] * pow(1-prob, i+1)
        eps_losses.append(loss[tgt].item())
    polyn_loss = torch.Tensor([i+j for i, j in zip(drop_poly, eps_losses)])

    if reduction == 'mean':
        return polyn_loss.mean()
    elif reduction == 'sum':
        return polyn_loss.sum()

    return polyn_loss


def polynfocal(probs, tgts, eps, gamma, n=100, reduction='none'):
    focaln = focallossfirstn(probs, tgts, gamma, n, 'none')
    eps_losses = list()
    for prob, tgt in zip(probs, tgts):
        loss = 0.0
        for i in range(len(eps)):
            loss += eps[i] * pow(1-prob, gamma+i+1)
        eps_losses.append(loss[tgt].item())
    polyn_focal = torch.Tensor([i+j for i, j in zip(focaln, eps_losses)])

    if reduction == 'mean':
        return polyn_focal.mean()
    elif reduction == 'sum':
        return polyn_focal.sum()

    return polyn_focal
