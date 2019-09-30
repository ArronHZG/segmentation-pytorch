import os
import time
from collections import namedtuple

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.datasets.utils import make_data_loader
from experiments.option import Options
from experiments.utils.saver import Saver
from experiments.utils.summaries import TensorboardSummary
from experiments.utils.tools import AverageMeter
from foundation import get_model, get_optimizer
from foundation.blseg.metric import metric
from progress.bar import Bar as Bar

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class Trainer:

    def __init__(self, args):
        self.args = args

        self.start_epoch = 1

        self.epochs = self.args.epochs

        self.best_pred = 0

        self.Metric = namedtuple('Metric', 'pixacc miou kappa')

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)

        # Define Dataloader
        train_set, val_set, self.class_num = make_data_loader(
            dataset_name=self.args.dataset,
            base_size=self.args.base_size,
            crop_size=self.args.crop_size,
        )
        self.in_c = train_set.in_c

        train_sampler = None
        val_sampler = None

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

        self.train_loader = DataLoader(train_set,
                                       batch_size=self.args.batch_size,
                                       shuffle=(train_sampler is None),
                                       pin_memory=True,
                                       num_workers=self.args.num_workers,
                                       sampler=train_sampler)
        self.val_loader = DataLoader(val_set,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=self.args.num_workers,
                                     sampler=val_sampler)

        # Define network
        print(f"=> creating model '{self.args.model}'", end=": ")
        self.model = get_model(model_name=self.args.model,
                               backbone=self.args.backbone,
                               num_classes=self.class_num,
                               in_c=self.in_c)
        print('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        print(f"=> creating optimizer '{self.args.optim}'")
        self.optimizer = get_optimizer(optim_name=self.args.optim, parameters=self.model.parameters(),
                                       lr=self.args.lr)

        if self.args.check_point_id is not None and self.args.experiment_dir_existed is True:
            print(f"=> reload  parameter from experiment_{self.args.check_point_id}")
            self.best_pred, self.start_epoch, model_state_dict, optimizer_state_dict = self.saver.load_checkpoint()
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        # self.criterion = loss.CrossEntropyLossWithOHEM( 0.7 )
        print(f"=> creating criterion 'CrossEntropyLoss'")
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        print(f"=> creating scheduler 'ReduceLROnPlateau'")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                                    factor=0.8,
                                                                    patience=3,
                                                                    verbose=True)

        if args.sync_bn:
            print("=> using apex synced BN")
            self.model = apex.parallel.convert_syncbn_model(self.model)

        print("\n=> apex\n")
        self.model = self.model.cuda()
        self.criterion.cuda()

        # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
        # for convenient interoperation with argparse.
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level=self.args.opt_level,
                                                    keep_batchnorm_fp32=self.args.keep_batchnorm_fp32,
                                                    loss_scale=self.args.loss_scale
                                                    )

        # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
        # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
        # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
        # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
        if args.distributed:
            # By default, apex.parallel.DistributedDataParallel overlaps communication with
            # computation in the backward pass.
            # model = DDP(model)
            # delay_allreduce delays all communication to the end of the backward pass.
            self.model = DDP(self.model, delay_allreduce=True)

        self.train_metric = self.Metric(miou=metric.MeanIoU(self.class_num),
                                        pixacc=metric.PixelAccuracy(),
                                        kappa=metric.Kappa(self.class_num))

        self.valid_metric = self.Metric(miou=metric.MeanIoU(self.class_num),
                                        pixacc=metric.PixelAccuracy(),
                                        kappa=metric.Kappa(self.class_num))

    def training(self, epoch):

        self.train_metric.miou.reset()
        self.train_metric.kappa.reset()
        self.train_metric.pixacc.reset()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        num_img_tr = len(self.train_loader)
        bar = Bar('Processing', max=num_img_tr)

        self.model.train()

        for batch_idx, sample in enumerate(self.train_loader):

            data_time.update(time.time() - end)

            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.optimizer.zero_grad()

            output = self.model(image)

            loss = self.criterion(output, target)

            self.train_metric.miou.update(output, target)
            self.train_metric.kappa.update(output, target)
            self.train_metric.pixacc.update(output, target)

            if self.args.apex:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # if hasattr(torch.cuda, 'empty_cache'):
            #     torch.cuda.empty_cache()

            self.optimizer.step()
            losses.update(loss.item())
            self.summary.writer.add_scalar('total_loss_iter', losses.avg,
                                           batch_idx + num_img_tr * epoch)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.train_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                mIoU=self.train_metric.miou.get(),
                Acc=self.train_metric.pixacc.get(),
            )
            bar.next()
        bar.finish()

        self.summary.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], epoch)
        self.summary.writer.add_scalars('metric/loss_epoch', {"train": losses.avg}, epoch)
        self.summary.writer.add_scalars('metric/mIoU', {"train": self.train_metric.miou.get()}, epoch)
        self.summary.writer.add_scalars('metric/Acc', {"train": self.train_metric.pixacc.get()}, epoch)
        self.summary.writer.add_scalars('metric/kappa', {"train": self.train_metric.kappa.get()}, epoch)

        print('[Epoch: %d, numImages: %5d]' % (epoch, num_img_tr * self.args.batch_size))
        print('Train Loss: %.3f' % losses.avg)

    def validation(self, epoch):

        self.valid_metric.miou.reset()
        self.valid_metric.kappa.reset()
        self.valid_metric.pixacc.reset()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        num_img_tr = len(self.val_loader)
        bar = Bar('Processing', max=num_img_tr)

        self.model.eval()

        for batch_idx, sample in enumerate(self.train_loader):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            losses.update(loss.item())
            self.valid_metric.miou.update(output, target)
            self.valid_metric.kappa.update(output, target)
            self.valid_metric.pixacc.update(output, target)
            # visualize_batch_image(image,
            #                       target, output, epoch, i,
            #                       self.saver.experiment_dir)

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.train_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                mIoU=self.valid_metric.miou.get(),
                Acc=self.valid_metric.pixacc.get(),
            )
            bar.next()
        bar.finish()

        # Fast test during the training
        new_pred = self.valid_metric.miou.get()
        metric_str = "Acc:{:.4f}, mIoU:{:.4f}, kappa: {:.4f}".format(self.valid_metric.pixacc.get(), new_pred,
                                                                     self.valid_metric.kappa.get())
        self.summary.writer.add_scalars('metric/loss_epoch', {"valid": losses.avg}, epoch)
        self.summary.writer.add_scalars('metric/mIoU', {"valid": new_pred}, epoch)
        self.summary.writer.add_scalars('metric/Acc', {"valid": self.valid_metric.pixacc.get()}, epoch)
        self.summary.writer.add_scalars('metric/kappa', {"valid": self.valid_metric.kappa.get()}, epoch)
        print('Validation:')
        print(f"[Epoch: {epoch}, numImages: {num_img_tr * self.args.batch_size}]")
        print(f'Valid Loss: {losses.avg:.4f}')

        is_best = new_pred > self.best_pred
        self.best_pred = max(new_pred, self.best_pred)

        if self.args.local_rank is 0:
            self.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, metric_str)
        return new_pred

    def auto_reset_learning_rate(self):
        if self.optimizer.param_groups[0]['lr'] <= 3e-4:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def train():
    args = Options().parse()
    trainer = Trainer(args)

    print("==> Start training")
    print('Total Epoches:', trainer.epochs)
    print('Starting Epoch:', trainer.start_epoch)
    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.training(epoch)
        if not args.no_val:
            new_pred = trainer.validation(epoch)
            trainer.scheduler.step(new_pred)
            trainer.auto_reset_learning_rate()


train()
