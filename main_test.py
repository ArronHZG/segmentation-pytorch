import gc
import os
import time
from glob import glob
from pprint import pprint

import torch
from tqdm import tqdm

from torch_model import get_model
from train_model.config.mypath import Path
from train_model.config.option import Options
from train_model.dataloader.rssrai_tools.rssrai_test import save_path, RssraiTestOneImage
from train_model.utils.saver import Saver


class Tester:

    def __init__(self):
        self.args = args
        self.saver = Saver(args)
        self.best_pred = 0
        self.start_epoch = 0
        self.model = None

    def test(self, image_name, image_dir, save_dir):
        rssraiImage = RssraiTestOneImage(image_name, image_dir, save_dir, self.args.batch_size, self.args.workers)

        # Define network
        self.model = get_model(model_name=self.args.model,
                               backbone=self.args.backbone,
                               num_classes=rssraiImage.num_classes,
                               in_c=rssraiImage.in_c)

        if self.args.check_point_id is not None:
            self.best_pred, self.start_epoch, model_state_dict, optimizer_state_dict = self.saver.load_checkpoint()
            self.model.load_state_dict(model_state_dict)
            del self.best_pred
            del self.start_epoch
            del optimizer_state_dict
            gc.collect()
        else:
            raise ModuleNotFoundError("can not find model file")

        if self.args.cuda:
            self.model = self.model.cuda()

        self.model.eval()

        typeList = ["origin", "vertical", "horizontal"]
        typeList = ["origin"]
        for type in typeList:
            for dateLoader in rssraiImage.get_slide_dataSet(rssraiImage.images[type]):
                tbar = tqdm(dateLoader)
                for i, sample in enumerate(tbar):
                    image = sample['image']
                    if self.args.cuda:
                        image = image.cuda()
                    with torch.no_grad():
                        sample['image'] = self.model(image)
                    rssraiImage.fill_image(type, sample)
                    del image
                    gc.collect()
                del tbar
                del dateLoader
                gc.collect()
            del rssraiImage.images[type]
            gc.collect()
        rssraiImage.saveResultRGBImage()

        del rssraiImage
        del self.model
        gc.collect()


if __name__ == "__main__":
    args = Options().parse()

    args.dataset = 'rssrai'
    args.model = 'FCN'
    args.backbone = 'resnet50'
    args.check_point_id = 1
    args.batch_size = 500

    print(args)
    now = f"{time.localtime(time.time()).tm_year}-{time.localtime(time.time()).tm_mon}-{time.localtime(time.time()).tm_mday}-"
    now += f"{time.localtime(time.time()).tm_hour}-{time.localtime(time.time()).tm_min}-{time.localtime(time.time()).tm_sec}"
    tester = Tester()

    base_dir = Path.db_root_dir('rssrai')
    image_dir = os.path.join(base_dir, 'train', 'img')
    save_dir = save_path(os.path.join(base_dir, f'test_output_model={args.model}_time={now}'))

    _img_path_list = glob(os.path.join(image_dir, '*.tif'))
    img_name_list = [name.split('/')[-1] for name in _img_path_list]
    pprint(img_name_list)
    for index, name in enumerate(img_name_list):
        print(f"{index}:{name}")
        tester.test(name, image_dir, save_dir)
