# @Time : 2023/2/24 16:36 
# @Author : Li Jiaqi
# @Description :
import torch
import torch.nn as nn
import torch.nn.functional as F


def diceCoeff(pred, gt, smooth_value: float = 1.0, activation=None):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1)
    fn = torch.sum(gt_flat, dim=1)
    loss = (2 * tp + smooth_value) / (fp + fn + smooth_value)
    return torch.mean(loss)


class SegmentationLoss(nn.Module):
    __name__ = 'seg_loss'

    def __init__(self, num_classes, activation=None, loss_type='dice'):
        super(SegmentationLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes
        self.loss_type = loss_type

    def forward(self, y_pred, y_true):
        class_loss = []
        for i in range(1, self.num_classes):
            if self.loss_type == 'dice':
                class_loss.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
            elif self.loss_type == 'cross_entropy':
                class_loss.append(F.cross_entropy(y_pred, y_true.float(), ignore_index=-1))
            else:
                raise "err lose type"
        mean_loss = sum(class_loss) / len(class_loss)
        return 1 - mean_loss
