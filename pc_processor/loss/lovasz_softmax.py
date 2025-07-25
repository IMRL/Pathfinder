"""

MIT License

Copyright (c) 2018 Maxim Berman
Copyright (c) 2020 Tiago Cortinhal, George Tzelepis and Eren Erdal Aksoy


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

"""
import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):

    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        # if v.numel() != 1:
        #     continue
        # else:
        #     acc += v
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=True, ignore=None):

    if per_image:
        losses = []
        for prob, lab in zip(probas, labels):
            prob_unsqueezed = prob.unsqueeze(0)
            lab_unsqueezed = lab.unsqueeze(0)
            flattened_probas, flattened_labels = flatten_probas(prob_unsqueezed, lab_unsqueezed, ignore)
            loss_value = lovasz_softmax_flat(flattened_probas, flattened_labels, classes=classes)
            if loss_value.numel() == 1:
                losses.append(loss_value)
            else:
                print("invalid lovasz loss_value")
        if losses:
            loss = mean(losses)
        else:
            loss = torch.tensor(0.0)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):

    if probas.numel() == 0:
        # raise ValueError('probas.numel() == 0')
        return probas * 0.

    if probas.dim() == 1:
        probas = probas.unsqueeze(0)
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output with multiple classes, but classes is not "all" or "present"')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        # losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
        lovasz_grad_values = lovasz_grad(fg_sorted)
        lovasz_grad_variable = Variable(lovasz_grad_values)
        dot_product = torch.dot(errors_sorted, lovasz_grad_variable)
        if dot_product.numel() != 1  :
            raise ValueError('dot_product error: ' + str(dot_product.numel()))
        losses.append(dot_product)
    losses_mean = mean(losses)
    return losses_mean


def flatten_probas(probas, labels, ignore=None):
    # print("input_label & input_imgmask in lovasz_1", labels.unique(return_counts=True))
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)

    if ignore is None:
        return probas, labels
    if isinstance(ignore, list):
        valid = (labels != ignore[0])
        for i in range(1, len(ignore)):
            valid = valid * (labels != ignore[i])
    else:
        valid = (labels != ignore)
    vprobas = probas[torch.nonzero(valid, as_tuple=False).squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', per_image=True, ignore=None):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)
