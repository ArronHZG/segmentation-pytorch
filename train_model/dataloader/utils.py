import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_model.dataloader.rssrai_tools import rssrai


def make_data_loader(args, **kwargs):
    if args.dataset == 'rssrai':
        train_set = rssrai.Rssrai(type='train',crop_size=args.crop_size)
        val_set = rssrai.Rssrai(type='valid')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
