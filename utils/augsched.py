import os
import torch
import yaml
import argparse
import numpy as np

# Augmentation Scheduler
# AugScheduler is a class that implements a scheduler for augmentation parameters
# It is used to gradually increase the strength of augmentation parameters
# It reads the augmentation parameters from a yaml file, which contains the maximum value of each parameter
# There are four modes of augmentation scheduling:
# 1. Cosine: the augmentation parameters are decreased following a half of a cosine curve
# 2. Sine: the augmentation parameters are increased following a half of a sine curve
# 3. full cosine: the augmentation parameters are decreased and increased following a full period of a cosine curve
# 4. full sine: the augmentation parameters are increased and decreased following a a full period of a sine curve
# The augmentation parameters are updated every epoch

class AugScheduler:
    def __init__(self, hyps, mode, max_epochs, aug_keys=None):
        self.hyps = hyps
        self.mode = mode
        self.max_epochs = max_epochs
        self.aug_keys = aug_keys

    def cosine(self, epoch):
        augs = self.hyps.copy()
        for k, v in augs.items():
            if k in self.aug_keys:
                augs[k] = v * (1 + torch.cos(torch.tensor(epoch * 3.141592 / self.max_epochs))) / 2
        return {k: v for k, v in augs.items()}

    def sine(self, epoch):
        augs = self.hyps.copy()
        for k, v in augs.items():
            if k in self.aug_keys:
                augs[k] = v * (1 - torch.cos(torch.tensor(epoch * 3.141592 / self.max_epochs))) / 2
        return {k: v for k, v in augs.items()}
        
    def full_cosine(self, epoch):
        augs = self.hyps.copy()
        for k, v in augs.items():
            if k in self.aug_keys:
                augs[k] = v * (1 + torch.cos(torch.tensor(epoch * 3.141592 / self.max_epochs / 2))) / 2
        return {k: v for k, v in augs.items()}

    def full_sine(self, epoch):
        augs = self.hyps.copy()
        for k, v in augs.items():
            if k in self.aug_keys:
                augs[k] = v * (1 - torch.cos(torch.tensor(epoch * 3.141592 / self.max_epochs / 2))) / 2
        return {k: v for k, v in augs.items()}

    def update(self, epoch):
        if self.mode == 'cosine':
            return self.cosine(epoch)
        elif self.mode == 'sine':
            return self.sine(epoch)
        elif self.mode == 'full_cosine':
            return self.full_cosine(epoch)
        elif self.mode == 'full_sine':
            return self.full_sine(epoch)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_file', type=str, default='augmentations.yaml')
    parser.add_argument('--augmode', type=str, default='cosine')
    parser.add_argument('--max_epochs', type=int, default=200)
    args = parser.parse_args()

    with open(args.aug_file, 'r') as f:
        hyps = yaml.load(f, Loader=yaml.FullLoader)

    aug = AugScheduler(hyps, args.augmode, args.max_epochs)
    for epoch in range(args.max_epochs):
        print(aug.update(epoch))

