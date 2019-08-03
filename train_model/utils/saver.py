import os
import shutil
import torch
from collections import OrderedDict
import glob


class Saver():

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.model + "-" + args.backbone )
        self.runs = glob.glob( os.path.join( self.directory, 'experiment_*' ) )
        run_ids = sorted( [int( experiment.split( '_' )[-1] ) for experiment in self.runs] ) if self.runs else [0]
        run_id = run_ids[-1] + 1

        print( f"run_id  {run_id}" )
        self.experiment_dir = os.path.join( self.directory, 'experiment_{}'.format( str( run_id ) ) )
        if not os.path.exists( self.experiment_dir ):
            os.makedirs( self.experiment_dir )

    def save_checkpoint(self, state, is_best, metric, filename='checkpoint.pth'):
        """Saves checkpoint to disk"""

        with open( os.path.join( self.experiment_dir, 'best_pred.txt' ), 'w+' ) as f:
            f.write( str( metric ) )
        if is_best:
            filename = os.path.join( self.experiment_dir, filename )
            torch.save( state, filename )

    def load_checkpoint(self):
        pass

        # 当你想恢复某一阶段的训练（或者进行测试）时，那么就可以读取之前保存的网络模型参数等。
        #
        # checkpoint = torch.load( dir )
        #
        # model.load_state_dict( checkpoint['net'] )
        #
        # optimizer.load_state_dict( checkpoint['optimizer'] )
        #
        # start_epoch = checkpoint['epoch'] + 1


    def save_experiment_config(self):
        logfile = os.path.join( self.experiment_dir, 'parameters.txt' )
        log_file = open( logfile, 'w' )
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss'] = self.args.loss
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write( key + ':' + str( val ) + '\n' )
        log_file.close()
