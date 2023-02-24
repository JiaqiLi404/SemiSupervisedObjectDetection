import utils.Register as Register
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
import torch
from collections import OrderedDict
import numpy as np
import logging

# logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)

LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)


def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls) * label, new_cls)
    return new_cls


class LoveDA(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list = []
        if isinstance(image_dir, list) and isinstance(mask_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        elif isinstance(image_dir, list) and not isinstance(mask_dir, list):
            for img_dir_path in image_dir:
                self.batch_generate(img_dir_path, mask_dir)
        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms

    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        logging.info('%s -- Dataset images: %d' % (os.path.dirname(image_dir), len(rgb_filepath_list)))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        if len(self.cls_filepath_list) > 0:
            mask = imread(self.cls_filepath_list[idx]).astype(np.long) - 1
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']
                mask = np.array(mask)
                image = np.array(image)
            individual_mask = []
            ones = np.ones(shape=mask.shape)
            zeros = np.zeros(shape=mask.shape)
            for i in LABEL_MAP.values():
                individual_mask.append(np.where(mask == i, ones, zeros))
            individual_mask = np.stack(individual_mask, 0)
            return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]), ind_mask=individual_mask)
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']
                image = np.array(image)

            return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)


@Register.DataLoaderRegister.register
class LoveDALoader(DataLoader):
    def __init__(self, config, flag="train"):
        self.config = config
        dataset = LoveDA(self.config["image_dir"][flag], self.config["mask_dir"][flag], self.config["transforms"])
        super(LoveDALoader, self).__init__(dataset,
                                           batch_size=self.config['batch_size'],
                                           num_workers=self.config['num_workers'],
                                           shuffle=self.config['shuffle'],
                                           pin_memory=self.config['pin_memory'],
                                           drop_last=self.config['drop_last']
                                           )
