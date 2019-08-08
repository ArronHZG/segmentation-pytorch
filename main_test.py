import os

import torch
from apex import amp
from tqdm import tqdm

from torch_model import get_model, get_optimizer
from train_model.config.option import Options
from train_model.dataloader.rssrai_tools.split_rssrai import merge_rssrai_test_label_images
from train_model.dataloader.utils import make_data_loader
from train_model.utils.saver import Saver
from train_model.utils.summaries import TensorboardSummary


class Tester():

    def __init__(self):
        self.args = args

        # Define Dataloader
        _, _, self.test_loader, self.class_num, self.val_dataset = make_data_loader(
            dataset_name=self.args.dataset,
            base_size=(self.args.base_size, self.args.base_size),
            crop_size=(self.args.crop_size, self.args.crop_size),
            batch_size=self.args.batch_size,
            num_workers=self.args.workers
        )

        # Define Saver
        self.saver = Saver(args)

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir, self.val_dataset)

        # Define network
        self.model = get_model(model_name=self.args.model,
                               backbone=self.args.backbone,
                               num_classes=self.class_num,
                               in_c=self.val_dataset.in_c)

        self.optimizer = get_optimizer(optim_name=self.args.optim, parameters=self.model.parameters(),
                                       lr=self.args.lr)

        if self.args.check_point_id != None:
            self.best_pred, self.start_epoch, model_state_dict, optimizer_state_dict = self.saver.load_checkpoint()
            self.model.load_state_dict(model_state_dict)

        if self.args.cuda:
            self.model = self.model.cuda()

        if self.args.apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=f"O{args.apex}")

    def test(self):
        self.model.eval()

        tbar = tqdm(self.test_loader)
        print(len(tbar))
        for i, sample in enumerate(tbar):
            image = sample['image']
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output = self.model(image)
            self.summary.save_test_image(sample['name'], output)
            # break

        self.summary.merge()


if __name__ == "__main__":
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    args = Options().parse()

    args.dataset = 'rssrai'
    args.model = 'DeepLabV3Plus'
    args.backbone = 'selu_se_resnet50'
    args.check_point_id = 17
    args.batch_size = 100
    args.base_size = 256
    args.crop_size = 256
    args.optim = "Adam"
    args.apex = 2
    args.epochs = 300
    # args.lr=0.01

    print(args)
    tester = Tester()
    tester.test()
