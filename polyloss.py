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
        num_classes = input.shape[1]
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


class PolyFocalLoss(_Loss):
    def __init__(self, epsilon: float = 1.0,
                 gamma: float = 2.0,
                 alpha: list[float] = None,
                 onehot_encoded: bool = False,
                 reduction: str = "mean",
                 weight: Optional[torch.Tensor] = None,
                 pos_weight: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the instance of PolyFocalLoss
        :param epsilon: scaling factor for leading polynomial coefficient
        :param gamma: exponent of the modulating factor (1 - p_t) to balance
                      easy vs hard examples.
        :param alpha: weighting factor per class in range (0,1) to balance
                      positive vs negative examples.
        :param onehot_encoded: True if target is one hot encoded.
        :param reduction: the reduction to apply to the output
                          'none': no reduction will be applied,
                          'mean': the weighted mean of the output is taken,
                          'sum': the output will be summed.
        """
        super().__init__()
        self.eps = epsilon
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, list):
                raise ValueError("Expected list of floats between 0-1"
                                 " for each class or None.")
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = alpha
        self.reduction = reduction
        self.onehot_encoded = onehot_encoded

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: output of neural network tensor of shape (n, num_classes)
                      or (n, num_classes, ...)
        :param target: ground truth tensor of shape (n, ) or (n, ...)
        :return: polyfocalloss
        """
        num_classes = input.shape[1]
        if not self.onehot_encoded:
            if target.ndim == 1:
                target1 = F.one_hot(target, num_classes=num_classes)
            else:
                # target is of size (n, ...)
                target1 = target.unsqueeze(1)  # (n, 1, ...)
                # (n, 1, ...) => (n, 1, ... , num_classes)
                target1 = F.one_hot(target1, num_classes=num_classes)
                # (n, 1, ..., num_classes) => (n, num_classes, ..., 1)
                target1 = target1.transpose(1, -1)
                # (n, num_classes, ..., 1) => (n, num_classes, ...)
                target1 = target1.squeeze(-1)

        target1 = target1.to(device=input.device, dtype=input.dtype)

        ce_loss =\
            F.cross_entropy(input, target1, reduction="none")

        p_t = torch.exp(-ce_loss)
        loss = torch.pow((1 - p_t), self.gamma) * ce_loss

        if self.alpha is not None:
            if len(self.alpha) != num_classes:
                raise ValueError("Alpha value is not available"
                                 " for all the classes.")
            if torch.count_nonzero(self.alpha) == 0:
                raise ValueError("All values can't be 0.")
            self.alpha = self.alpha/sum(self.alpha)
            alpha_t = self.alpha.gather(0, target.data.view(-1))
            loss *= alpha_t
            poly_loss = loss + self.eps * torch.pow(1-p_t,
                                                    self.gamma+1) * alpha_t
        else:
            poly_loss = loss + self.eps * torch.pow(1-p_t, self.gamma+1)

        if self.reduction == "mean":
            poly_loss = poly_loss.mean()
        elif self.reduction == "sum":
            poly_loss = poly_loss.sum()

        return poly_loss
