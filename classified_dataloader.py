import random
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
import numpy as np


class SitesClassified(Dataset):
    def __init__(self, data_dir, category, transforms=None):
        self.data_dir = data_dir + '\\' + category
        print(self.data_dir)
        self.id_list = []
        self.category = category
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
        image = imread(os.path.join(self.data_dir, file_name))
        image = image[:-23, :, 0:3]  # delete the alpha dimension in png files and bing flag

        # normalize and resize
        if self.transforms is None:
            blob = self.transforms(image=image)
            image = blob['image']

        # to C,W,H
        image = np.rollaxis(image, 2, 0)
        return image

    def __len__(self):
        return len(self.id_list)


class SitesLoader(DataLoader):
    def __init__(self, config, dataset=None, flag="labeled"):
        self.config = config
        self.flag = flag
        if dataset:
            pass
        elif flag == 'labeled':
            dataset = SitesClassified(self.config["dataset"], self.config["category"], self.config["transforms"])
        elif flag == 'unlabeled':
            dataset = SitesClassified(self.config["unlabeledset"], self.config["category"], self.config["transforms"])
        super(SitesLoader, self).__init__(dataset,
                                          batch_size=self.config['batch_size'],
                                          num_workers=self.config['num_workers'],
                                          shuffle=self.config['shuffle'],
                                          pin_memory=self.config['pin_memory'],
                                          drop_last=self.config['drop_last']
                                          )

    def reshuffle(self):
        random.shuffle(SitesLoader.dataset.id_list)

