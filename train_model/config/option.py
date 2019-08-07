import argparse

import torch


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset
        parser.add_argument('--model', type=str, default='DeepLabV3Plus',
                            help='model name (default: DeepLabV3Plus)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='rssrai',
                            help='dataset name (default: rssrai)')
        parser.add_argument('--workers', type=int, default=12,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=513,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=512,
                            help='crop image size')
        parser.add_argument('--pretrained', action='store_true', default=False,
                            help='model name (default: False)')
        parser.add_argument('--pretrained-weight-path', type=str, default=None,
                            help='model name (default: None)')
        # training hyper params

        # parser.add_argument('--aux', action='store_true', default= False,
        #                     help='Auxilary Loss')
        # parser.add_argument('--se-loss', action='store_true', default= False,
        #                     help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=64,
                            metavar='N', help='input batch size for training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--optim', type=str, default="SGD",
                            help='(default: auto)')
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--lr-step', type=int, default=None,
                            help='lr step to change lr')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        # apex  a Pytorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training
        parser.add_argument('--apex', type=int, default=2, choices=[0, 1, 2, 3], help='Automatic Mixed Precision')

        # # finetuning pre-trained model
        # parser.add_argument('--ft', action='store_true', default= False,
        #                     help='finetuning on a different dataset')
        # parser.add_argument('--ft-resume', type=str, default=None,
        #                     help='put the path of trained model to finetune if needed ')
        # parser.add_argument('--pre-class', type=int, default=None,
        #                     help='num of pre-trained classes \
        #                     (default: None)')

        # evaluation option
        parser.add_argument('--ema', action='store_true', default=False,
                            help='using EMA evaluation')
        parser.add_argument('--eval', action='store_true', default=False,
                            help='evaluating mIoU')
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')
        # # test option
        # parser.add_argument('--test-folder', type=str, default=None,
        #                     help='path to test image folder')
        # parser.add_argument('--multi-scales',action="store_true", default=False,
        #                     help="testing scale,default:1.0(single scale)")
        # # multi grid dilation option
        # parser.add_argument("--multi-grid", action="store_true", default=False,
        #                     help="use multi grid dilation policy")
        # parser.add_argument('--multi-dilation', nargs='+', type=int, default=None,
        #                     help="multi grid dilation list")
        # parser.add_argument('--scale', action='store_false', default=True,
        #                    help='choose to use random scale transform(0.75-2),default:multi scale')

        # check_point
        parser.add_argument('--check-point-id', type=int, default=None)

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'pascal_voc': 50,
                'pascal_aug': 50,
                'pcontext': 80,
                'ade20k': 160,
                'cityscapes': 180,
                'rssrai': 100,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.batch_size is None:
            args.batch_size = 4 * torch.cuda.device_count()
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.lr is None:
            lrs = {
                'pascal_voc': 0.1,
                'pascal_aug': 0.1,
                'pcontext': 0.1,
                'ade20k': 0.1,
                'cityscapes': 0.1,
                'rssrai': 0.1,
            }
            args.lr = lrs[args.dataset.lower()] / 256 * args.batch_size
        return args
