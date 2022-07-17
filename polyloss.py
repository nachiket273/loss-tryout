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
        :param label_smoothing: the amount of smoothening applied to labels.
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
        :param input: output of neural network tensor of shape (n, num_classes)
        :param target: ground truth tensor of shape (n, )
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
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()

        return poly_loss


# focal_loss implementation based on implementation:
# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
class PolyFocalLoss(_Loss):
    def __init__(self, epsilon: float = 1.0,
                 gamma: float = 2.0,
                 alpha: float = -1,
                 onehot_encoded: bool = False,
                 reduction: str = "mean",
                 weight: Optional[torch.Tensor] = None,
                 pos_weight: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the instance of PolyFocalLoss
        :param epsilon: scaling factor for leading polynomial coefficient
        :param gamma: exponent of the modulating factor (1 - p_t) to balance
                      easy vs hard examples.
        :param alpha: weighting factor in range (0,1) to balance positive vs
                      negative examples.
        :param onehot_encoded: True if target is one hot encoded.
        :param reduction: the reduction to apply to the output
                          'none': no reduction will be applied,
                          'mean': the weighted mean of the output is taken,
                          'sum': the output will be summed.
        :param weight: manual rescaling weight provided to binary cross entropy
                       loss.
        :param pos_weight: weight of positive examples provided to binary
                           cross entropy loss.
        """
        super().__init__()
        self.eps = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.onehot_encoded = onehot_encoded
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: output of neural network tensor of shape (n, num_classes)
                      or (n, num_classes, ...)
        :param target: ground truth tensor of shape (n, ) or (n, ...)
        :return: polyfocalloss
        """
        p = torch.sigmoid(input)
        num_classes = input.shape[1]

        if not self.onehot_encoded:
            if target.ndim == 1:
                target = F.one_hot(target, num_classes=num_classes)
            else:
                # target is of size (n, ...)
                target = target.unsqueeze(1)  # (n, 1, ...)
                # (n, 1, ...) => (n, 1, ... , num_classes)
                target = F.one_hot(target, num_classes=num_classes)
                # (n, 1, ..., num_classes) => (n, num_classes, ..., 1)
                target = target.transpose(1, -1)
                # (n, num_classes, ..., 1) => (n, num_classes, ...)
                target = target.squeeze(-1)

        target = target.to(device=input.device, dtype=input.dtype)

        ce_loss =\
            F.binary_cross_entropy_with_logits(input, target,
                                               weight=self.weight,
                                               pos_weight=self.pos_weight,
                                               reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        poly_loss = loss + self.eps * torch.pow(1 - p_t,
                                                self.gamma + 1) * alpha_t

        if self.reduction == "mean":
            poly_loss = poly_loss.mean()
        elif self.reduction == "sum":
            poly_loss = poly_loss.sum()

        return poly_loss
