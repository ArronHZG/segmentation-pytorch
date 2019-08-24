from collections import namedtuple

import torch
from apex import amp
from tqdm import tqdm

from torch_model import get_model, get_optimizer
from torch_model.blseg.loss import loss
from torch_model.blseg.metric import metric
from train_model.config.option import Options
from train_model.dataloader.utils import make_data_loader
from train_model.utils.saver import Saver
from train_model.utils.summaries import TensorboardSummary


class Trainer():

    def __init__(self):
        self.args = args

        self.start_epoch = self.args.start_epoch

        self.epochs = self.args.epochs

        self.best_pred = 0

        # Define Dataloader
        self.train_loader, self.val_loader, self.test_loader, self.class_num, self.val_dataset = make_data_loader(
            dataset_name=self.args.dataset,
            base_size=(self.args.base_size, self.args.base_size),
            crop_size=(self.args.crop_size, self.args.crop_size),
            batch_size=self.args.batch_size,
            num_workers=self.args.workers
        )

        # Define Saver
        self.saver = Saver( args )
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary( self.saver.experiment_dir, self.val_dataset )

        # Define network
        self.model = get_model( model_name=self.args.model,
                                backbone=self.args.backbone,
                                num_classes=self.class_num,
                                in_c=self.val_dataset.in_c)

        self.optimizer = get_optimizer( optim_name=self.args.optim, parameters=self.model.parameters(),
                                        lr=self.args.lr )

        if self.args.check_point_id != None:
            self.best_pred, self.start_epoch, model_state_dict, optimizer_state_dict = self.saver.load_checkpoint()
            self.model.load_state_dict( model_state_dict )
            self.optimizer.load_state_dict( optimizer_state_dict )

        # self.criterion = loss.CrossEntropyLossWithOHEM( 0.7 )
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( self.optimizer, 'max',
                                                                     factor=0.9,
                                                                     patience=3,
                                                                     verbose=True )
        if self.args.cuda:
            self.model = self.model.cuda()
            self.criterion.cuda()
        if self.args.apex:
            self.model, self.optimizer = amp.initialize( self.model, self.optimizer, opt_level=f"O{args.apex}" )

        self.Metric = namedtuple( 'Metric', 'pixacc miou kappa' )

        self.train_metric = self.Metric( miou=metric.MeanIoU( self.class_num ),
                                         pixacc=metric.PixelAccuracy(),
                                         kappa=metric.Kappa( self.class_num ) )

        self.valid_metric = self.Metric( miou=metric.MeanIoU( self.class_num ),
                                         pixacc=metric.PixelAccuracy(),
                                         kappa=metric.Kappa( self.class_num ) )

    def training(self, epoch):

        self.train_metric.miou.reset()
        self.train_metric.kappa.reset()
        self.train_metric.pixacc.reset()
        train_loss = 0.0
        self.model.train()
        tbar = tqdm( self.train_loader )
        num_img_tr = len( self.train_loader )
        for i, sample in enumerate( tbar ):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.optimizer.zero_grad()

            output = self.model( image )

            loss = self.criterion( output, target )

            self.train_metric.miou.update( output, target )
            self.train_metric.kappa.update( output, target )
            self.train_metric.pixacc.update( output, target )

            if self.args.apex:
                with amp.scale_loss( loss, self.optimizer ) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # if hasattr(torch.cuda, 'empty_cache'):
            #     torch.cuda.empty_cache()

            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description( 'Train loss: %.3f' % (train_loss / (i + 1)) )
            self.summary.writer.add_scalar( 'total_loss_iter', train_loss / (i + 1), i + num_img_tr * epoch )

        train_loss /= num_img_tr
        self.summary.writer.add_scalar( "learning_rate", self.optimizer.param_groups[0]['lr'], epoch )
        self.summary.writer.add_scalars( 'metric/loss_epoch', {"train": train_loss}, epoch )
        self.summary.writer.add_scalars( 'metric/mIoU', {"train": self.train_metric.miou.get()}, epoch )
        self.summary.writer.add_scalars( 'metric/Acc', {"train": self.train_metric.pixacc.get()}, epoch )
        self.summary.writer.add_scalars( 'metric/kappa', {"train": self.train_metric.kappa.get()}, epoch )

        print( '[Epoch: %d, numImages: %5d]' % (epoch, num_img_tr * self.args.batch_size) )
        print( 'Loss: %.3f' % train_loss )

    def validation(self, epoch):
        self.model.eval()

        self.valid_metric.miou.reset()
        self.valid_metric.kappa.reset()
        self.valid_metric.pixacc.reset()

        tbar = tqdm( self.val_loader )
        num_img_tr = len( tbar )
        test_loss = 0.0
        for i, sample in enumerate( tbar ):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model( image )
            loss = self.criterion( output, target )
            test_loss += loss.item()
            tbar.set_description( 'Test loss: %.3f' % (test_loss / (i + 1)) )
            self.valid_metric.miou.update( output, target )
            self.valid_metric.kappa.update( output, target )
            self.valid_metric.pixacc.update( output, target )
            self.summary.visualize_image( image, target, output, epoch ,i)

        # Fast test during the training

        new_pred = self.valid_metric.miou.get()
        test_loss /= num_img_tr
        metric_str = f"Acc:{self.valid_metric.pixacc.get()}, mIoU:{new_pred}, kappa: {self.valid_metric.kappa.get()}"
        self.summary.writer.add_scalars( 'metric/loss_epoch', {"valid": test_loss}, epoch )
        self.summary.writer.add_scalars( 'metric/mIoU', {"valid": new_pred}, epoch )
        self.summary.writer.add_scalars( 'metric/Acc', {"valid": self.valid_metric.pixacc.get()}, epoch )
        self.summary.writer.add_scalars( 'metric/kappa', {"valid": self.valid_metric.kappa.get()}, epoch )
        print( 'Validation:' )
        print( '[Epoch: %d, numImages: %5d]' % (epoch, num_img_tr * self.args.batch_size) )
        print(metric_str)
        print( 'Loss: %.3f' % test_loss )

        is_best = False

        if new_pred > self.best_pred:
            self.best_pred = new_pred
            is_best = True

        self.saver.save_checkpoint( {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best, metric_str )

        return new_pred


if __name__ == "__main__":

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    args = Options().parse()


    args.dataset = 'rssrai'
    args.model = 'FCN'
    args.backbone = 'resnet101'
    # args.check_point_id = 1
    args.batch_size = 70
    args.base_size = 256
    args.crop_size = 256
    args.optim = "SGD"
    args.apex = 2
    args.epochs=500
    args.lr=0.01


    print( args )
    trainer = Trainer()
    print( 'Total Epoches:', trainer.epochs )
    print( 'Starting Epoch:', trainer.start_epoch )
    for epoch in range( trainer.start_epoch, trainer.epochs ):
        trainer.training( epoch )
        if not args.no_val:
            new_pred = trainer.validation( epoch )
            trainer.scheduler.step( new_pred )
