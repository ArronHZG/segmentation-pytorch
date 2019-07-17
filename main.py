from dataloader.utils import make_data_loader
from models.utils import get_model
from utils.option import Options
from utils.saver import Saver
from utils.summaries import TensorboardSummary


class Trainer():
    def __init__(self):
        self.args = args

        # Define Saver
        self.saver = Saver( args )
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.class_num = make_data_loader(args, **kwargs)


        # Define network
        self.model = get_model(model_name=self.args.model,
                               num_classes=self.class_num,
                               backbone=self.args.backbone,
                               pretrained=self.args.pretrained,
                               pretrained_weight_path=self.args.pretrained_weight_path)

    def training(self, epoch):
        pass

    def validation(self, epoch):
        pass


if __name__ == "__main__":
    args = Options().parse()
    print(args)
    trainer = Trainer()
    print( 'Total Epoches:', args.epochs )
    print( 'Starting Epoch:', args.start_epoch )
    for epoch in range( args.start_epoch, args.epochs ):
        trainer.training( epoch )
        if not args.no_val:
            trainer.validation( epoch )
