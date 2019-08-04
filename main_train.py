from collections import namedtuple

import torch
from apex import amp
from tqdm import tqdm

from torch_model import get_model
from torch_model.blseg.loss import loss
from torch_model.blseg.metric import metric
from train_model.config.option import Options
from train_model.dataloader.utils import make_data_loader
from train_model.utils.saver import Saver
from train_model.utils.summaries import TensorboardSummary


class Trainer():

    def __init__(self):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir, self.args.dataset)

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}

        self.train_loader, self.val_loader, self.test_loader, self.class_num = make_data_loader(args, **kwargs)

        # Define network
        # self.model = get_model(model_name=self.args.model,
        #                        num_classes=self.class_num,
        #                        backbone=self.args.backbone,
        #                        pretrained=self.args.pretrained,
        #                        pretrained_weight_path=self.args.pretrained_weight_path)

        self.start_epoch = self.args.start_epoch

        self.epochs = self.args.epochs

        self.best_pred = 0

        self.model = get_model(model_name=self.args.model, backbone=self.args.backbone, num_classes=self.class_num)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        if self.args.check_point_id != None:
            self.best_pred, self.start_epoch, model_state_dict, optimizer_state_dict = self.saver.load_checkpoint()
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.criterion = loss.CrossEntropyLossWithOHEM(0.7)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    factor=0.9,
                                                                    patience=3,
                                                                    verbose=True)
        if self.args.cuda:
            self.model = self.model.cuda()

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=f"O{args.apex}")

        self.pixacc = metric.PixelAccuracy()

        self.miou = metric.MeanIoU(self.class_num)

        self.kappa = metric.Kappa(self.class_num)

        self.Metric = namedtuple('Metric', 'pixacc miou kappa')

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.optimizer.zero_grad()

            output = self.model(image)

            loss = self.criterion(output, target)

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # if hasattr(torch.cuda, 'empty_cache'):
            #     torch.cuda.empty_cache()

            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.summary.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.summary.writer.add_scalar('train/loss_epoch', train_loss / len(tbar), epoch)
        self.summary.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch):
        self.model.eval()

        self.miou.reset()
        self.kappa.reset()
        self.pixacc.reset()

        tbar = tqdm(self.val_loader)
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            self.miou.update(output, target)
            self.kappa.update(output, target)
            self.pixacc.update(output, target)
            self.summary.visualize_image(image, target, output, epoch)

        metric = self.Metric(miou=self.miou.get(), pixacc=self.pixacc.get(), kappa=self.kappa.get())

        # Fast test during the training
        self.summary.writer.add_scalar('val/loss_epoch', test_loss / len(tbar), epoch)
        self.summary.writer.add_scalar('val/mIoU', metric.miou, epoch)
        self.summary.writer.add_scalar('val/Acc', metric.pixacc, epoch)
        self.summary.writer.add_scalar('val/kappa', metric.kappa, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print(f"Acc:{metric.pixacc}, mIoU:{metric.miou}, kappa: {metric.kappa}")
        print('Loss: %.3f' % test_loss)

        new_pred = metric.miou
        is_best = False

        if new_pred > self.best_pred:
            self.best_pred = new_pred
            is_best = True

        self.saver.save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best, metric)

        return new_pred


if __name__ == "__main__":
    args = Options().parse()

    args.check_point_id = 1
    args.model = 'DeepLabV3Plus'
    args.batch_size = 64
    args.crop_size = 256
    print(args)
    trainer = Trainer()
    print('Total Epoches:', trainer.epochs)
    print('Starting Epoch:', trainer.start_epoch)
    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.training(epoch)
        if not args.no_val:
            new_pred = trainer.validation(epoch)
            trainer.scheduler.step(new_pred)
