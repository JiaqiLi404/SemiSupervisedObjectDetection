import random

import numpy
import torch
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
import numpy as np
import config


def get_categories(flag='labeled'):
    if flag == 'unlabeled':
        path = os.listdir(config.DataLoaderConfig["unlabeledcalssified"])
    else:
        path = os.listdir(config.DataLoaderConfig["labeledcalssified"])
    return path


class SitesClassified(Dataset):
    def __init__(self, data_dir, category, mask_dir, transforms=None):
        self.data_dir = os.path.join(data_dir, category)
        self.id_list = []
        self.category = category
        self.mask_dir = mask_dir
        self.unlabeled = False

        png_list = glob.glob(os.path.join(self.data_dir, '*.png'))
        for fp in png_list:
            if len(os.path.split(fp)[-1]) > 8:
                self.id_list.append(os.path.split(fp)[-1][:-8])
            else:
                self.id_list.append(os.path.split(fp)[-1][:-4])
                self.unlabeled = True
        self.transforms = transforms



    def __getitem__(self, idx):
        file_id = self.id_list[idx]
        file_name = file_id + 'bing.png' if not self.unlabeled else file_id + '.png'
        mask_name = file_id + 'bing_mask.png'
        image = imread(os.path.join(self.data_dir, file_name))
        image = image[:-23, :, 0:3]  # delete the alpha dimension in png files and bing flag
        mask_image = []
        if not self.unlabeled:
            mask_image = imread(os.path.join(self.mask_dir, mask_name))
            mask_image = mask_image[:-23, :, 0:3]  # delete the alpha dimension in png file
        # normalize and resize
        if self.transforms is not None:
            if not self.unlabeled:
                mask_image = mask_image[:, :, 0]
                blob = self.transforms(image=image, mask=mask_image)
                image = blob['image']
                mask_image = blob['mask']
                mask_image = (mask_image - np.min(mask_image)) / (
                        np.max(mask_image) - np.min(mask_image))
            else:
                blob = self.transforms(image=image)
                image = blob['image']

        # to C,W,H
        image = np.rollaxis(image, 2, 0)
        return image, mask_image

    def __len__(self):
        return len(self.id_list)


class SitesLoader(DataLoader):
    def __init__(self, config, category, flag="labeled"):
        self.config = config
        self.flag = flag
        self.category = category
        if category not in get_categories(flag):
            assert "category not found"
        if flag == 'labeled':
            dataset = SitesClassified(self.config["labeledcalssified"], category, self.config["maskdir"],
                                      self.config["transforms"])
        elif flag == 'unlabeled':
            dataset = SitesClassified(self.config["unlabeledcalssified"], category, None,
                                      self.config["transforms"])
        super(SitesLoader, self).__init__(dataset,
                                          batch_size=self.config['few_shot_batch_size'],
                                          num_workers=self.config['num_workers'],
                                          shuffle=self.config['shuffle'],
                                          pin_memory=self.config['pin_memory'],
                                          drop_last=self.config['drop_last']
                                          )
    def reshuffle(self):
        random.shuffle(self.dataset.id_list)

