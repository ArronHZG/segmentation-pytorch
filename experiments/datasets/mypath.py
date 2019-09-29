class Path():
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/arron/Documents/arron/d2l-zh/data/VOCdevkit/VOC2012'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'  # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/home/arron/Documents/arron/dataSet/coco'
        elif dataset == 'rssrai':
            return '/home/arron/Documents/grey/Project_Rssrai/rssrai'
        else:
            print( 'Dataset {} not available.'.format( dataset ) )
            raise NotImplementedError

    @staticmethod
    def pretrain_models_root_dir(model):
        if model == "resnet18":
            return '/home/arron/PycharmProjects/segmentation-pytorch/pretrain_models/resnet18-5c106cde.pth'
        elif model == "resnet34":
            return '/home/arron/PycharmProjects/segmentation-pytorch/pretrain_models/resnet34-333f7ec4.pth'
        elif model == "resnet50":
            return '/home/arron/PycharmProjects/segmentation-pytorch/pretrain_models/resnet50-19c8e357.pth'
        elif model == "resnet101":
            return '/home/arron/PycharmProjects/segmentation-pytorch/pretrain_models/resnet101-5d3b4d8f.pth'
        elif model == "resnet152":
            return '/home/arron/PycharmProjects/segmentation-pytorch/pretrain_models/resnet152-b121ed2d.pth'
        else:
            print( 'Model {} not available.'.format( model ) )
            raise NotImplementedError

    project_root = '/home/arron/PycharmProjects/segmentation-pytorch'
