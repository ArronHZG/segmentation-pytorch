# cython: language_level=3
import glob
import os
from collections import OrderedDict

import torch

from experiments.utils.iotools import make_sure_path_exists


class Saver:

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.model + "-" + args.backbone)
        if None is self.args.check_point_id:
            self.runs = glob.glob(os.path.join(self.directory, 'experiment_*'))
            run_ids = sorted([int(experiment.split('_')[-1]) for experiment in self.runs]) if self.runs else [0]
            run_id = run_ids[-1] + 1
            self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
            make_sure_path_exists(self.experiment_dir)
        else:
            run_id = self.args.check_point_id
            self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
            if not os.path.exists(self.experiment_dir):
                raise RuntimeError(f"{self.experiment_dir} doesn't exist")

    def save_checkpoint(self, state, is_best, metric):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, 'reality_checkpoint.pth')
        torch.save(state, filename)
        with open(os.path.join(self.experiment_dir, 'train_log.txt'), 'a+') as f:
            f.write(f"[epoch {str(state['epoch']).zfill(3)}] {metric}\n")
        if is_best:
            filename = os.path.join(self.experiment_dir, 'best_checkpoint.pth')
            torch.save(state, filename)
            with open(os.path.join(self.experiment_dir, 'best_log.txt'), 'a+') as f:
                f.write(f"[epoch {str(state['epoch']).zfill(3)}] {metric}\n")

    def load_checkpoint(self, is_best=False):
        # 当你想恢复某一阶段的训练（或者进行测试）时，那么就可以读取之前保存的网络模型参数等。
        try:
            if is_best:
                checkpoint = torch.load(os.path.join(self.experiment_dir, 'best_checkpoint.pth'))
            else:
                checkpoint = torch.load(os.path.join(self.experiment_dir, 'reality_checkpoint.pth'))
        except Exception:
            raise RuntimeError("checkpoint doesn't exist.")
        return checkpoint

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
        p['batch_size'] = self.args.batch_size
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write(f"\n\n{'=' * 20}\n")
        log_file.write(str(self.args))
        log_file.write(f"\n")
        log_file.close()
