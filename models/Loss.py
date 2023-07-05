# @Time : 2023/2/24 16:36 
# @Author : Li Jiaqi
# @Description :
import torch
import torch.nn as nn
import torch.nn.functional as F

''' this function is used to calculate the dice coefficient, which shows the similarity of two samples
parameter pred: predicted class
            gt: ground truth
  smooth_value:
    activation: activation function
'''


def diceCoeff(pred, gt, smooth_value: float = 1.0, activation="softmax2d"):
    """ computational formula:
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    elif activation == "softmax":
        activation_fn = nn.Softmax()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.abs(torch.sum(gt_flat * pred_flat, dim=1))
    fp = torch.sum(torch.abs(pred_flat), dim=1)
    fn = torch.sum(gt_flat, dim=1)
    score = (2 * tp + smooth_value) / (fp + fn + smooth_value)
    return torch.mean(score)


def mse(predicted, gt):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    N = gt.size(0)
    gt_flat = gt.reshape(N, -1)
    pred_flat = predicted.reshape(N, -1)
    err = torch.sum((gt_flat - pred_flat) ** 2, 1)
    err /= (gt.shape[0] * gt.shape[1])

    return torch.mean(err)


class SegmentationLoss(nn.Module):
    __name__ = 'seg_loss'

    def __init__(self, num_classes, activation=None, loss_type='dice'):
        super(SegmentationLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes
        self.loss_type = loss_type

    def forward(self, y_pred, y_true):
        ##store the similarity score of the prediction and the true value
        if len(y_pred.shape) == 3:
            y_pred = y_pred.unsqueeze(dim=1)
        if len(y_true.shape) == 3:
            y_true = y_true.unsqueeze(dim=1)
        if self.loss_type == 'mse':
            return mse(y_pred, y_true)
        class_score = []
        for i in range(0, self.num_classes):
            if self.loss_type == 'dice':
                class_score.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :], activation=self.activation))
            elif self.loss_type == 'cross_entropy':
                class_score.append(F.cross_entropy(y_pred, y_true.float(), ignore_index=-1))
            else:
                raise "err lose type"
        ##?Question: it should be count the dice loss first like 1-dice_score then caculate the mean or 
        ##                                                     score it first then mean
        mean_loss = sum(class_score) / len(class_score)
        return 1 - mean_loss
