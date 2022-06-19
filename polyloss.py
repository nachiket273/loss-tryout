import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Optional

__all__ = [
    "PolyLoss",
    "PolyFocalLoss"
]


class PolyLoss(_Loss):
    def __init__(self,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 label_smoothing: float = 0.) -> None:
        """
        Initialize the instance of PolyLoss.
        :param ce_weight: manual rescaling weight given to each class.
        :param reduction: the reduction to apply to the output
                          'none': no reduction will be applied,
                          'mean': the weighted mean of the output is taken,
                          'sum': the output will be summed.
        :param epsilon: scaling factor for leading polynomial coefficient
        :param label_smoothing: the amount of smoothing applied to labels.
        """
        super().__init__()
        self.reduction = reduction
        self.eps = epsilon
        self.ce_weight = ce_weight
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: float tensor of shape (n, num_classes)
        :param target: long tensor of shape (n, )
        :return: polyloss
        """
        num_classes = input.shape[-1]
        labels_onehot = F.one_hot(target, num_classes=num_classes)
        labels_onehot = labels_onehot.to(device=input.device,
                                         dtype=input.dtype)

        if self.label_smoothing > 0.0:
            smooth_labels = labels_onehot * (1 - self.label_smoothing)\
                + self.label_smoothing / num_classes
            ce = F.cross_entropy(input, target,
                                 weight=self.ce_weight,
                                 reduction='none',
                                 label_smoothing=self.label_smoothing)

            one_minus_pt = torch.sum(smooth_labels * (1 - F.softmax(input,
                                                                    dim=-1)),
                                     dim=-1)
            poly_loss = ce + self.eps * one_minus_pt
        else:
            pt = torch.sum(labels_onehot * F.softmax(input, dim=-1), dim=-1)
            ce = F.cross_entropy(input, target,
                                 weight=self.ce_weight,
                                 reduction='none')
            poly_loss = ce + self.eps * (1 - pt)

        if self.reduction == 'mean':
            return torch.mean(poly_loss, dim=-1)
        elif self.reduction == 'sum':
            return torch.sum(poly_loss, dim=-1)

        return poly_loss
