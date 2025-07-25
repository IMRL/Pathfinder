import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalSoftmaxLossImg(nn.Module):
    def __init__(self, n_classes, gamma=1, beta=0.8, softmax=True, signal=None):
        super(FocalSoftmaxLossImg, self).__init__()
        self.gamma = gamma
        self.n_classes = 2
        self.signal = signal

        if isinstance(beta, list):
            assert len(beta) == n_classes, "len(beta)!=n_classes: {} vs. {}".format(
                len(beta), n_classes)
            self.beta = torch.Tensor(beta)
        elif isinstance(beta, np.ndarray):
            assert beta.shape[0] == n_classes, "len(beta)!=n_classes: {} vs. {}".format(
                len(beta), n_classes)
            self.beta = torch.from_numpy(beta)
        else:
            assert beta < 1 and beta > 0, "invalid beta: {}".format(beta)
            self.beta = torch.zeros(n_classes)
            self.beta[0] = beta
            self.beta[1:] += (1-beta)
        self.softmax = softmax

    def forward(self, x, target, mask=None, signal=None):
        """compute focal loss
        x: N C or NCHW
        target: N, or NHW

        Args:
            x ([type]): [description]
            target ([type]): [description]
        """

        if signal >= 9:
            self.gamma = 0.0
            self.beta[0] = 1.0
            self.beta[1] = 1.0

        if x.dim() > 2:
            pred = x.view(x.size(0), x.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, x.size(1))
        else:
            pred = x

        target = target.view(-1, 1)

        if self.softmax:
            pred_softmax = F.softmax(pred, 1)
        else:
            pred_softmax = pred
        pred_softmax = pred_softmax.gather(1, target).view(-1)
        pred_logsoft = pred_softmax.clamp(1e-6).log()
        self.beta = self.beta.to(x.device)
        beta = self.beta.gather(0, target.squeeze())
        loss = - (1-pred_softmax).pow(self.gamma)
        loss = loss * pred_logsoft * beta
        if mask is not None:
            if len(mask.size()) > 1:
                mask = mask.view(-1)
            loss = (loss * mask).sum() / mask.sum() if mask.sum() > 0 else (loss * mask).sum()
            # loss = (loss * mask).sum() / mask.sum()
            # print("loss 1", loss)
            if torch.isnan(loss):
              print("Focal input", torch.isnan(x).sum().item())
              print("Focal pred_softmax", pred_softmax)
              print("Focal pred_softmax item= Nan",  torch.isnan(pred_softmax).sum().item())
              print("Focal pred_logsoft", pred_logsoft)
              print("Focal pred_logsoft item= Nan",  torch.isnan(pred_logsoft).sum().item())
              print("Focal beta", beta)
              print("Focal target 0",  (target == 0).sum().item())
              print("Focal target 1",  (target == 1).sum().item())

            return loss
        else:
            # print("loss 2", loss)
            return loss.mean()


if __name__ == "__main__":
    criterion = FocalSoftmaxLossImg(n_classes=2, gamma=1, beta=0.8)
    target = torch.arange(0, 10)
    print(target)
    test_input = torch.rand(10, 10)
    mask = torch.ones(10)
    mask[4] = 0
    loss = criterion(test_input, target, mask)
    print(loss)
