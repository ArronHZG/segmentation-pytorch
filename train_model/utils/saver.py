import glob
import os
from collections import OrderedDict

import torch


class Saver:

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.model + "-" + args.backbone)
        if None is not self.args.check_point_id:
            self.runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
            run_ids = sorted([int(experiment.split('_')[-1]) for experiment in self.runs]) if self.runs else [0]
            run_id = run_ids[-1] + 1
            self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)
        else:
            run_id = self.args.check_point_id
            self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
            if not os.path.exists(self.experiment_dir):
                raise FileNotFoundError(self.experiment_dir)
        print(f"run_id  {run_id}")

    def save_checkpoint(self, state, is_best, metric, filename='checkpoint.pth'):
        """Saves checkpoint to disk"""
        with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'a+') as f:
            f.write(f"[epoch {str(state['epoch']).zfill(3)}] {metric}\n")
        if is_best:
            filename = os.path.join(self.experiment_dir, filename)
            torch.save(state, filename)

    def load_checkpoint(self):
        # 当你想恢复某一阶段的训练（或者进行测试）时，那么就可以读取之前保存的网络模型参数等。
        checkpoint = torch.load(os.path.join(self.experiment_dir, 'checkpoint.pth'))
        state_dict = checkpoint['state_dict']
        optimizer = checkpoint['optimizer']
        best_pred = checkpoint['best_pred']
        start_epoch = checkpoint['epoch'] + 1
        return best_pred, start_epoch, state_dict, optimizer

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['optim'] = self.args.optim
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
