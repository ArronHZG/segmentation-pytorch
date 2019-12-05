import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image
from progress.bar import Bar as Bar
from torch.utils.data import DataLoader

from experiments.datasets.utils import make_data_loader, decode_segmap
from experiments.option import Options
from experiments.utils.iotools import make_sure_path_exists
from experiments.utils.saver import Saver
from foundation import get_model, get_optimizer
from foundation.metric import MeanIoU, PixelAccuracy, Kappa, AverageMeter

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class Message:
    def __init__(self, **kwargs):
        for item in kwargs.items():
            setattr(self, item[0], item[1])


class Trainer:

    def __init__(self, args):
        self.args = args

        self.start_epoch = 1

        self.epochs = self.args.epochs

        self.best_pred = 0

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        if self.args.tensorboard:
            from experiments.utils.summaries import TensorboardSummary
            make_sure_path_exists(os.path.join(self.saver.experiment_dir, "tensorboard"))
            self.summary = TensorboardSummary(os.path.join(self.saver.experiment_dir, "tensorboard"))

        # Define Dataloader
        train_set, val_set, self.num_classes = make_data_loader(
            dataset_name=self.args.dataset,
            base_size=self.args.base_size,
            crop_size=self.args.crop_size,
            basic_dir=self.args.basic_dir
        )
        self.in_c = train_set.in_c
        self.mean = train_set.mean
        self.std = train_set.std

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
                                     batch_size=self.args.val_batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=self.args.num_workers,
                                     sampler=val_sampler)

        # Define network
        print(f"=> creating model '{self.args.model}'", end=": ")
        self.model = get_model(model_name=self.args.model,
                               backbone=self.args.backbone,
                               num_classes=self.num_classes,
                               in_c=self.in_c).cuda()

        print('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        print(f"=> creating optimizer '{self.args.optim}'")
        self.optimizer = get_optimizer(optim_name=self.args.optim, parameters=self.model.parameters(),
                                       lr=self.args.lr)

        if self.args.check_point_id is not None:
            print(f"=> reload  parameter from experiment_{self.args.check_point_id}")
            checkpoint = self.saver.load_checkpoint()
            self.best_pred = checkpoint['best_pred']
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # amp.load_state_dict(checkpoint['amp'])

        # self.criterion = loss.CrossEntropyLossWithOHEM( 0.7 )
        print(f"=> creating criterion 'CrossEntropyLoss'")
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()

        print(f"=> creating scheduler 'ReduceLROnPlateau'")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                                    factor=0.8,
                                                                    patience=3,
                                                                    verbose=True)

        if args.sync_bn:
            print("=> using apex synced BN")
            self.model = apex.parallel.convert_syncbn_model(self.model)

        print("=> using apex")
        # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
        # for convenient interoperation with argparse.
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level=self.args.opt_level,
                                                    keep_batchnorm_fp32=self.args.keep_batchnorm_fp32,
                                                    loss_scale=self.args.loss_scale)

        # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
        # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
        # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
        # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
        # if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        # self.model = DDP(self.model, device_ids=self.args.gpu_ids, delay_allreduce=True)

        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=self.args.gpu_ids)

        id_list = [int(id) for id in args.gpu_ids.split(',')]
        print(f"using gpus {id_list}")

        if len(id_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=id_list)

        self.train_message = Message(miou=MeanIoU(self.num_classes),
                                     pixacc=PixelAccuracy(),
                                     kappa=Kappa(self.num_classes),
                                     batch_time=AverageMeter(),
                                     data_time=AverageMeter(),
                                     loss=AverageMeter(),
                                     lr=self.optimizer.param_groups[0]['lr'],
                                     total_time=0)

        self.val_message = Message(miou=MeanIoU(self.num_classes),
                                   pixacc=PixelAccuracy(),
                                   kappa=Kappa(self.num_classes),
                                   batch_time=AverageMeter(),
                                   data_time=AverageMeter(),
                                   loss=AverageMeter(),
                                   total_time=0)

    def training(self, epoch):

        self.train_message.miou.reset()
        self.train_message.pixacc.reset()
        self.train_message.kappa.reset()
        self.train_message.batch_time.reset()
        self.train_message.data_time.reset()
        self.train_message.loss.reset()

        batch_num = len(self.train_loader)
        bar = Bar('train', max=batch_num)
        self.model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        for batch_idx, sample in enumerate(self.train_loader):

            image, target = sample['image'], sample['label']
            self.train_message.data_time.update(time.time() - batch_start_time)

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.optimizer.zero_grad()

            output = self.model(image)

            loss = self.criterion(output, target)

            self.train_message.miou.update(output, target)
            self.train_message.kappa.update(output, target)
            self.train_message.pixacc.update(output, target)
            self.train_message.loss.update(loss.item())

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # if hasattr(torch.cuda, 'empty_cache'):
            #     torch.cuda.empty_cache()

            self.optimizer.step()
            if self.args.tensorboard:
                self.summary.writer.add_scalar('total_loss_iter', self.train_message.loss.avg,
                                           batch_idx + batch_num * epoch)

            # measure elapsed time
            self.train_message.batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # plot progress
            bar.suffix = '[{epoch}]({batch}/{size}) lr: {lr:.4f} | Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f}'.format(
                epoch=epoch,
                batch=batch_idx + 1,
                size=batch_num,
                data=self.train_message.data_time.avg,
                bt=self.train_message.batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                lr=self.train_message.lr,
                loss=self.train_message.loss.avg,
                mIoU=self.train_message.miou.get(),
                Acc=self.train_message.pixacc.get(),
            )
            bar.next()
        bar.finish()

        # calculate total time
        self.train_message.total_time = time.time() - epoch_start_time
        # get lr
        self.train_message.lr = self.optimizer.param_groups[0]['lr']

        if self.args.tensorboard:
            self.summary.writer.add_scalar("learning_rate", self.train_message.lr, epoch)
            self.summary.writer.add_scalars('metric/loss_epoch', {"train": self.train_message.loss.avg}, epoch)
            self.summary.writer.add_scalars('metric/mIoU', {"train": self.train_message.miou.get()}, epoch)
            self.summary.writer.add_scalars('metric/Acc', {"train": self.train_message.pixacc.get()}, epoch)
            self.summary.writer.add_scalars('metric/kappa', {"train": self.train_message.kappa.get()}, epoch)

        # print('[Epoch: %d, numImages: %5d]' % (epoch, batch_num * self.args.batch_size))

    def validation(self, epoch):

        self.val_message.miou.reset()
        self.val_message.kappa.reset()
        self.val_message.pixacc.reset()
        self.train_message.batch_time.reset()
        self.val_message.data_time.reset()
        self.val_message.loss.reset()

        batch_num = len(self.val_loader)
        bar = Bar('valid', max=batch_num)
        self.model.eval()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        for batch_idx, sample in enumerate(self.val_loader):

            image, target = sample['image'], sample['label']
            self.val_message.data_time.update(time.time() - batch_start_time)

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, target)

            self.val_message.miou.update(output, target)
            self.val_message.kappa.update(output, target)
            self.val_message.pixacc.update(output, target)
            self.val_message.loss.update(loss.item())

            # self.visualize_batch_image(image, target, output, epoch, batch_idx)

            # measure elapsed time
            self.val_message.batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # plot progress
            bar.suffix = '[{epoch}]({batch}/{size}) lr: {lr:.4f} | Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f}'.format(
                epoch=epoch,
                batch=batch_idx + 1,
                size=batch_num,
                data=self.val_message.data_time.avg,
                bt=self.val_message.batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                lr=self.train_message.lr,
                loss=self.val_message.loss.avg,
                mIoU=self.val_message.miou.get(),
                Acc=self.val_message.pixacc.get(),
            )
            bar.next()
        bar.finish()

        # calculate total time
        self.train_message.total_time = time.time() - epoch_start_time

        # Fast test during the training
        new_pred = self.val_message.miou.get()
        is_best = new_pred > self.best_pred
        self.best_pred = max(new_pred, self.best_pred)

        if self.args.tensorboard:
            self.summary.writer.add_scalars('metric/loss_epoch', {"valid": self.val_message.loss.avg}, epoch)
            self.summary.writer.add_scalars('metric/mIoU', {"valid": new_pred}, epoch)
            self.summary.writer.add_scalars('metric/Acc', {"valid": self.val_message.pixacc.get()}, epoch)
            self.summary.writer.add_scalars('metric/kappa', {"valid": self.val_message.kappa.get()}, epoch)

        save_message = f"train\t{self.message_str('train')}\t val\t{self.message_str('val')}"

        if self.args.local_rank is 0:
            self.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'amp': amp.state_dict()
            }, is_best, save_message)
        return new_pred

    def auto_reset_learning_rate(self):
        if self.optimizer.param_groups[0]['lr'] <= 1e-4:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def visualize_batch_image(self, image, target, output,
                              epoch, batch_index):
        # image (B,C,H,W) To (B,H,W,C)
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, axes=[0, 2, 3, 1])
        image_np *= self.std
        image_np += self.mean
        image_np *= 255.0
        image_np = image_np.astype(np.uint8)

        # target (B,H,W)
        target = target.cpu().numpy()

        # output (B,C,H,W) to (B,H,W)
        output = torch.argmax(output, dim=1).cpu().numpy()

        for i in range(min(3, image_np.shape[0])):
            img_tmp = image_np[i]
            img_rgb_tmp = np.array(Image.fromarray(img_tmp).convert("RGB")).astype(np.uint8)
            target_rgb_tmp = decode_segmap(target[i], self.num_classes).astype(np.uint8)
            output_rgb_tmp = decode_segmap(output[i], self.num_classes).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(131)
            plt.imshow(img_rgb_tmp, vmin=0, vmax=255)
            plt.subplot(132)
            plt.imshow(target_rgb_tmp, vmin=0, vmax=255)
            plt.subplot(133)
            plt.imshow(output_rgb_tmp, vmin=0, vmax=255)
            path = os.path.join(self.saver.experiment_dir, "vis_image", f'epoch_{epoch}')
            make_sure_path_exists(path)
            plt.savefig(f"{path}/{batch_index}-{i}.jpg")
            plt.close('all')

    def message_str(self, model):
        message = None
        string = ''

        if model is "train":
            message = self.train_message
            string = f"lr: {message.lr:.4f}, "
        elif model is "val":
            message = self.val_message

        return string + \
               f"total_time: {message.total_time:.4f}, " + \
               f"loss: {message.loss.avg:.4f}, " + \
               f"acc: {message.pixacc.get():.4f}, " + \
               f"kappa: {message.kappa.get():.4f}, " + \
               f"mIoU: {message.miou.get():.4f}, " + \
               f"item_mIoU: {[f'{x:.4f}' for x in message.miou.get_item()]}"


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


if __name__ == '__main__':
    train()
