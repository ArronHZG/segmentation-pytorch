import argparse
import os
from pprint import pprint

import torch.backends.cudnn as cudnn

import torch


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # net and datasets
        parser.add_argument('--model', type=str, help='net name', required=True)
        parser.add_argument('--dataset', type=str, help='dataset name', required=True)
        parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name (default: resnet50)')
        parser.add_argument('--num-workers', type=int, default=12, metavar='N', help='datasets threads')
        parser.add_argument('--base-size', type=int, default=520, help='base image size (default: 520)')
        parser.add_argument('--crop-size', type=int, default=480, help='crop image size (default: 480)')
        parser.add_argument('--pretrained', action='store_true', default=True, help='net name (default: True)')
        parser.add_argument('--check-point-id', type=int, default=None)

        # training hyper params
        parser.add_argument('--aux', action='store_true', default=False, help='Auxilary Loss')
        parser.add_argument('--se-loss', action='store_true', default=False, help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--epochs', type=int, default=None, metavar='N', help='default: auto')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='default: auto')
        parser.add_argument('--test-batch-size', type=int, default=None, metavar='N',
                            help='(default: same as batch size)')
        # optimizer params
        parser.add_argument('--optim', type=str, default="SGD", help='(default: SGD)')
        parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='ReduceLROnPlateau',
                            help='learning rate scheduler (default: ReduceLROnPlateau)')
        # apex
        # a Pytorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training
        parser.add_argument('--apex', type=int, default=2, choices=[0, 1, 2, 3], help='Automatic Mixed Precision')
        parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
        parser.add_argument('--loss-scale', type=str, default=None)

        # cuda
        parser.add_argument('--gpu-ids', type=str, required=True)
        parser.add_argument('--deterministic', action='store_false')
        parser.add_argument("--local_rank", default=0, type=int)
        parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')

        # # finetuning pre-trained net
        # parser.add_argument('--ft', action='store_true', default= False,
        #                     help='finetuning on a different dataset')
        # parser.add_argument('--ft-resume', type=str, default=None,
        #                     help='put the path of trained net to finetune if needed ')
        # parser.add_argument('--pre-class', type=int, default=None,
        #                     help='num of pre-trained classes \
        #                     (default: None)')

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

        # the parser
        self.parser = parser

    lrs = {
        'coco': 0.01,
        'citys': 0.01,
        'pascal_voc': 0.0001,
        'pascal_aug': 0.001,
        'pcontext': 0.001,
        'ade20k': 0.01,
        'rssrai': 0.01,
    }

    epoches = {
        'coco': 30,
        'citys': 240,
        'pascal_voc': 50,
        'pascal_aug': 50,
        'pcontext': 80,
        'ade20k': 120,
        'rssrai': 1000
    }

    def init_distributed_cuda(self, args):
        # pytorch如何能够保证模型的可重复性
        cudnn.benchmark = True
        if args.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.manual_seed(args.local_rank)
            torch.set_printoptions(precision=10)

        args.distributed = False
        args.gpu = 0
        args.world_size = 1

        if 'WORLD_SIZE' in os.environ:
            args.distributed = int(os.environ['WORLD_SIZE']) > 1
            print(f"world_size {int(os.environ['WORLD_SIZE'])}\n")
        print(f"\ndistributed {args.distributed}\n")

        if args.distributed:
            args.gpu = args.local_rank
            torch.cuda.set_device(args.gpu)
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            args.world_size = torch.distributed.get_world_size()

        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

        args.lr = args.lr * float(args.batch_size * args.world_size) / 256.

    def parse(self):
        args = self.parser.parse_args()

        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            args.epochs = self.epoches[args.dataset.lower()]

        if args.lr is None:
            args.lr = self.lrs[args.dataset.lower()]

        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size

        # Use CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        args.cuda = torch.cuda.is_available()

        # Use apex
        args.opt_level = f"O{args.apex}"

        args.experiment_dir_existed = False

        pprint(args)

        print(f"\nCUDNN VERSION: {torch.backends.cudnn.version()}\n")

        self.init_distributed_cuda(args)

        return args
