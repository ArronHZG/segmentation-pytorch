from torch.utils.data import DataLoader

from train_model.dataloader.rssrai_tools import rssrai
from train_model.dataloader.voc2012 import pascal


def make_data_loader(dataset_name, base_size, crop_size, batch_size, num_workers):
    if dataset_name == 'rssrai':
        train_set = rssrai.Rssrai( type='train', base_size=base_size, crop_size=crop_size )
        val_set = rssrai.Rssrai( type='valid', base_size=base_size, crop_size=crop_size )
        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader( train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                   num_workers=num_workers )
        val_loader = DataLoader( val_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers )
        test_loader = None

        dataset = val_set

        return train_loader, val_loader, test_loader, num_class, dataset

    if dataset_name == 'voc2012':

        train_set = pascal.VOCSegmentation( type='train', base_size=base_size, crop_size=crop_size )
        val_set = pascal.VOCSegmentation( type='val', base_size=base_size, crop_size=crop_size )
        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader( train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                   num_workers=num_workers )
        val_loader = DataLoader( val_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers )
        test_loader = None

        dataset = val_set

        return train_loader, val_loader, test_loader, num_class, dataset

    else:
        raise NotImplementedError
