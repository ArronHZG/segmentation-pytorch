import gc
import os
import sys
import time
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from experiments.datasets.private.testOneImage import TestOneImage
from experiments.option import Options
from experiments.utils.img_to_tif import parseIMG
from experiments.utils.numpy_to_vis_shp import numpy_to_shp, numpy_to_vis
from experiments.utils.saver import Saver
from foundation import get_model

np.set_printoptions(threshold=sys.maxsize)
Image.MAX_IMAGE_PIXELS = None

END_TIME = time.strptime("2020 3 20 20 09", "%Y %m %d %H %M")


def check_time():
    cur_time = time.localtime(time.time())

    e_time = int(time.mktime(END_TIME))
    c_time = int(time.mktime(cur_time))

    if e_time > c_time:
        return True
    return False


def save_resize_image(np_image, path, name, times=1):
    image = Image.fromarray(np_image.astype('uint8'))
    if times is not 1:
        h = np_image.shape[0]
        w = np_image.shape[1]
        image = image.resize((w // times, h // times), Image.ANTIALIAS)
    image.save(os.path.join(path, name))


class Tester:

    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.best_pred = 0
        self.start_epoch = 0

        self.num_classes = args.test_num_classes
        self.in_c = 4

        # Define network
        self.model = get_model(model_name=self.args.model,
                               backbone=self.args.backbone,
                               num_classes=self.num_classes,
                               in_c=self.in_c)

        # Resume weight

        if self.args.check_point_id is not None:

            print(f"=> reload  parameter from experiment_{self.args.check_point_id}")
            checkpoint = self.saver.load_checkpoint(is_best=True)
            self.best_pred = checkpoint['best_pred']
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            del checkpoint
            gc.collect()
        else:
            raise ModuleNotFoundError("can not find model file")
        self.model = self.model.cuda()
        self.model.eval()

    def test(self, image_dir, image_name):
        testOneImage = TestOneImage(self.num_classes, self.args.crop_size, os.path.join(image_dir, image_name),
                                    self.args.batch_size,
                                    self.args.num_workers)

        dateLoader = testOneImage.get_DataLoader()
        print(f"=> start test")
        for sample in tqdm(dateLoader):
            image = sample['image'].cuda()
            with torch.no_grad():
                sample['image'] = self.model(image)
            testOneImage.fill_image(sample)
            del image
            gc.collect()
        del dateLoader
        gc.collect()
        output = testOneImage.output_image
        result = np.zeros(output.shape[:2], dtype=np.int8)
        id_number = output.shape[0] // 1000
        remainder = output.shape[0] % 1000
        if remainder != 0:
            id_number += 1
        for id in range(id_number - 1):
            temp = torch.from_numpy(output[id * 1000:(id + 1) * 1000]).cuda()
            temp = torch.argmax(temp, -1)
            result[id * 1000:(id + 1) * 1000] = temp.cpu().numpy()
        temp = torch.from_numpy(output[(id_number - 1) * 1000:]).cuda()
        temp = torch.argmax(temp, -1)
        result[(id_number - 1) * 1000:] = temp.cpu().numpy()
        return result

        # output /= np.max(output)
        # output *= 255
        # output = output.astype(np.uint8)
        # img = Image.fromarray(output)
        # os.makedirs(save_dir, exist_ok=True)
        # img.save(os.path.join(save_dir, image_name))


def test():
    if not check_time():
        raise RuntimeError("Not Found train.lib")
    args = Options().parse()
    print(args)

    # 将原图转换为 tif
    # 将tif 转换为 numpy
    # 将numpy 转换为 shp 及可视化

    tester = Tester(args)

    basic_dir = args.basic_dir
    img_dir = os.path.join(basic_dir, 'img')
    tif_dir = os.path.join(basic_dir, 'result-tif')
    tif_vis_dir = os.path.join(basic_dir, 'result-tif-vis')
    numpy_dir = os.path.join(basic_dir, 'result-numpy')
    shp_dir = os.path.join(basic_dir, 'result-shp')
    result_vis_dir = os.path.join(basic_dir, 'result-vis')

    os.makedirs(tif_dir,exist_ok=True)
    os.makedirs(tif_vis_dir,exist_ok=True)
    os.makedirs(numpy_dir,exist_ok=True)
    os.makedirs(shp_dir,exist_ok=True)
    os.makedirs(result_vis_dir,exist_ok=True)

    img_path_list = glob(os.path.join(img_dir, '*.img'))
    name_list = [os.path.split(x)[-1] for x in img_path_list]
    code_list = [os.path.splitext(x)[0] for x in name_list]

    for index, code in enumerate(code_list):
        print(f'===> {index + 1}/{len(code_list)} {code}')
        tif_image = parseIMG(os.path.join(img_dir, f"{code}.img"))
        save_resize_image(tif_image.image, tif_dir, f"{code}.tif")
        save_resize_image(tif_image.image, tif_vis_dir, f"{code}.tif", times=8)
        del tif_image.image
        result = tester.test(tif_dir, f"{code}.tif")
        np.savez(os.path.join(numpy_dir, code), pred=result)
        numpy_to_shp(tester.num_classes, tif_image, numpy_dir, shp_dir, code)
        numpy_to_vis(tester.num_classes, numpy_dir, result_vis_dir, code)
