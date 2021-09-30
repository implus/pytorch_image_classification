import torch
import torch.nn.functional as F
import yacs.config
import math


def onehot_encoding(label: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)


def cross_entropy_loss(data: torch.Tensor, target: torch.Tensor,
                       reduction: str) -> torch.Tensor:
    logp = F.log_softmax(data, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')


class LabelDynamicHistoryLoss:
    PI = math.acos(-1.0)
    def __init__(self, config: yacs.config.CfgNode, reduction: str):
        self.n_classes = config.dataset.n_classes
        #self.epsilon = config.augmentation.label_smoothing.epsilon
        #self.momentum_label = config.label.momentum_label # 0.9
        self.momentum_label_final = config.label.momentum_label_final # 0.1
        self.momentum_label_range = -config.label.momentum_label_final + \
                                     config.label.momentum_label
        #self.momentum_history = config.label.momentum_history # 0.9
        self.momentum_history_final = config.label.momentum_history_final # 0.1
        self.momentum_history_range = -config.label.momentum_history_final + \
                                       config.label.momentum_history
        self.reduction = reduction
        self.total_epoch = config.scheduler.epochs # 200 cosine decay

    def __call__(self, predictions, targets, label_dict, step, idx, epoch):
        device = predictions.device

        onehot = onehot_encoding(
            targets, self.n_classes).type_as(predictions).to(device)

        prob = F.softmax(predictions.detach(), dim=1)
        #print('prob.size() = ', prob.size())

        if len(label_dict[step]) == 0:
            targets = onehot
            label_dict[step][idx] = prob.clone()
        else:
            ratio = 1.0 * epoch / self.total_epoch
            scale = (math.cos(ratio * self.PI) + 1.) / 2
            curr_momentum_label = scale * self.momentum_label_range + self.momentum_label_final
            curr_momentum_history = scale * self.momentum_history_range + self.momentum_history_final
            #print('curr momentum label', curr_momentum_label)
            #print('curr momentum history', curr_momentum_history)

            targets = onehot * curr_momentum_label + \
                label_dict[step][idx] * (1. - curr_momentum_label)
            # update history
            label_dict[step][idx] = prob * curr_momentum_history + \
                label_dict[step][idx] * (1. - curr_momentum_history)
        
        #targets = onehot * (1 - self.epsilon) + torch.ones_like(onehot).to(
        #    device) * self.epsilon / self.n_classes
        loss = cross_entropy_loss(predictions, targets, self.reduction)
        return loss
