import torch
import numpy as np


class PixelAccuracy:

    def __init__(self, ignore_index=-100, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0
        self.ignore_index = ignore_index
        self.eps = eps

    def update(self, pred, target):
        ignore_mask = target != self.ignore_index
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)
        self.num_correct += ((pred.long() == target.long()) *
                             ignore_mask).sum().item()
        self.num_instance += ignore_mask.sum().item()

    def get(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0


class MeanIoU:

    def __init__(self, num_classes, eps=1e-7):
        if num_classes == 1:
            self.num_classes = num_classes + 1
        else:
            self.num_classes = num_classes
        self.num_intersection = np.zeros(self.num_classes)
        self.num_union = np.zeros(self.num_classes)
        self.eps = eps

    def update(self, pred, target):
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)

        for cur_cls in range(self.num_classes):
            pred_mask = (pred == cur_cls).byte()
            target_mask = (target == cur_cls).byte()

            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()

            self.num_intersection[cur_cls] += intersection.item()
            self.num_union[cur_cls] += union.item()

    def get(self, ignore_background=False):
        if ignore_background:
            return (self.num_intersection[1:] /
                    (self.num_union[1:] + self.eps)).mean()
        else:
            return (self.num_intersection / (self.num_union + self.eps)).mean()

    def reset(self):
        self.num_intersection = np.zeros(self.num_classes)
        self.num_union = np.zeros(self.num_classes)


class Kappa:
    def __init__(self, num_classes):
        self.pre_vec = np.zeros(num_classes)
        self.cor_vec = np.zeros(num_classes)
        self.tar_vec = np.zeros(num_classes)
        self.num = num_classes

    def update(self, output, target):
        pre_array = torch.argmax(output, dim=1)

        for i in range(self.num):
            pre_mask = (pre_array == i).byte()
            tar_mask = (target == i).byte()
            self.cor_vec[i] = (pre_mask & tar_mask).sum().item()
            self.pre_vec[i] = pre_mask.sum().item()
            self.tar_vec[i] = tar_mask.sum().item()

    def get(self):
        assert len(self.pre_vec) == len(self.tar_vec) == len(self.pre_vec)
        tmp = 0.0
        for i in range(len(self.tar_vec)):
            tmp += self.pre_vec[i] * self.tar_vec[i]
        pe = tmp / (sum(self.tar_vec) ** 2 + 1e-8)
        p0 = sum(self.cor_vec) / (sum(self.tar_vec) + 1e-8)
        cohens_coefficient = (p0 - pe) / (1 - pe)
        return cohens_coefficient

    def reset(self):
        self.pre_vec = np.zeros(self.num)
        self.cor_vec = np.zeros(self.num)
        self.tar_vec = np.zeros(self.num)


def testKappa():
    out = torch.randn(2, 16, 256, 256).cuda()
    tar = torch.randint(16, (2, 256, 256)).cuda()
    kappa = Kappa(num_classes=16)
    kappa.update(out, tar)
    print(kappa.get())

    mean = MeanIoU(num_classes=16)
    mean.update(out, tar)
    print(mean.get())


    acc =PixelAccuracy()
    acc.update(out,tar)
    print(acc.get())



if __name__ == '__main__':
    testKappa()
