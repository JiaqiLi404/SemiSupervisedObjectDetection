# @Time : 2023/4/22 19:22 
# @Author : Li Jiaqi
# @Description :
from torch.utils.data import Dataset
import glob
import os
from skimage.transform import resize
from skimage.io import imread
from torch.utils.data import DataLoader
import torch
from collections import OrderedDict
import numpy as np
import logging

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


class SitesBingBook(Dataset):
    def __init__(self, data_dir, mask_dir, transforms=None, has_bing=True, has_book=False, has_mask=True):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.has_bing = has_bing
        self.has_book = has_book
        self.has_mask = has_mask
        self.id_list = []
        self.unlabeled=False
        png_list = glob.glob(os.path.join(data_dir, '*.png'))
        for fp in png_list:
            if 'mask' not in fp and len(os.path.split(fp)[-1])>8:
                self.id_list.append(os.path.split(fp)[-1][:-8])
            elif 'mask' not in fp:
                self.id_list.append(os.path.split(fp)[-1][:-4])
                self.unlabeled =True
        self.transforms = transforms
        self.feature_extractor = None

    def __getitem__(self, idx):
        file_id = self.id_list[idx]
        bing_file_name = file_id + 'bing.png' if not self.unlabeled else file_id + '.png'
        book_file_name = file_id + 'book.jpg'
        bing_mask_name = file_id + 'bing_mask.png'
        book_mask_name = file_id + 'book_mask.png'
        bing_image, bing_mask_image, book_image, book_mask_image = [], [], [], []
        if self.has_bing:
            bing_image = imread(os.path.join(self.data_dir, bing_file_name))
            bing_image = bing_image[:-23, :, 0:3]  # delete the alpha dimension in png files and bing flag
            if self.has_mask:
                bing_mask_image = imread(os.path.join(self.mask_dir, bing_mask_name))
                bing_mask_image = bing_mask_image[:-23, :, 0:3]  # delete the alpha dimension in png files and bing flag
        if self.has_book:
            book_image = imread(os.path.join(self.data_dir, book_file_name))
            book_image = book_image[:-75, :]  # delete the book flag
            if self.has_mask:
                book_mask_image = imread(os.path.join(self.mask_dir, book_mask_name))
                book_mask_image = book_mask_image[:-75, :]  # delete the book flag

        # 2d black and white book images to 3d images
        if self.has_book and len(book_image.shape) <= 2:
            new_book_image = np.zeros(dtype=np.uint8, shape=(book_image.shape[0], book_image.shape[1], 3))
            new_book_image[:, :, 0] = book_image * 255
            new_book_image[:, :, 1] = book_image * 255
            new_book_image[:, :, 2] = book_image * 255
            book_image = new_book_image
        elif self.has_book and book_image.dtype != np.uint8:
            book_image = book_image * 255

        # normalize and resize
        if self.transforms is not None:
            if self.has_mask:
                if self.has_bing:
                    bing_mask_image = bing_mask_image[:, :, 0]
                    blob = self.transforms(image=bing_image, mask=bing_mask_image)
                    bing_image = blob['image']
                    bing_mask_image = blob['mask']
                    bing_mask_image = (bing_mask_image - np.min(bing_mask_image)) / (
                                np.max(bing_mask_image) - np.min(bing_mask_image))
                if self.has_book:
                    book_mask_image = book_mask_image[:, :, 0]
                    blob = self.transforms(image=book_image, mask=book_mask_image)
                    book_image = blob['image']
                    book_mask_image = blob['mask']
                    book_mask_image = (book_mask_image - np.min(book_mask_image)) / (
                            np.max(book_mask_image) - np.min(book_mask_image))
            else:
                if self.has_bing:
                    blob = self.transforms(image=bing_image)
                    bing_image = blob['image']
                if self.has_book:
                    blob = self.transforms(image=book_image)
                    book_image = blob['image']

        if self.feature_extractor is not None:
            if self.has_mask:
                if self.has_bing:
                    print('before', bing_image.shape)
                    encoded_inputs = self.feature_extractor(bing_image, bing_mask_image, return_tensors="pt")
                    bing_image = encoded_inputs["pixel_values"][0]
                    print('after', bing_image.shape)
                    bing_mask_image = encoded_inputs["labels"][0]
                if self.has_book:
                    encoded_inputs = self.feature_extractor(book_image, book_mask_image, return_tensors="pt")
                    book_image = encoded_inputs["pixel_values"][0]
                    book_mask_image = encoded_inputs["labels"][0]
            else:
                if self.has_bing:
                    encoded_inputs = self.feature_extractor(bing_image, return_tensors="pt")
                    bing_image = encoded_inputs["pixel_values"][0]
                if self.has_book:
                    encoded_inputs = self.feature_extractor(book_image, return_tensors="pt")
                    book_image = encoded_inputs["pixel_values"][0]
        else: 
            # to C,W,H
            if self.has_bing:
                bing_image = np.rollaxis(bing_image, 2, 0)
            if self.has_book:
                book_image = np.rollaxis(book_image, 2, 0)

        return bing_image, bing_mask_image, book_image, book_mask_image

    def __len__(self):
        return len(self.id_list)


class SitesLoader(DataLoader):
    def __init__(self, config, dataset=None, flag="train"):
        self.config = config
        self.flag = flag
        if dataset:
            pass
        elif flag == 'train':
            dataset = SitesBingBook(self.config["dataset"], self.config["maskdir"], self.config["transforms"])
        elif flag == 'unlabeled':
            dataset = SitesBingBook(self.config["unlabeledset"], None, self.config["transforms"], has_mask=False)
        elif flag=='pseudo':
            dataset = SitesBingBook(self.config["pseudoset"], None, self.config["transforms"], has_mask=False)
        else:
            dataset = SitesBingBook(self.config["evalset"], self.config["maskdir"], self.config["transforms"])
        super(SitesLoader, self).__init__(dataset,
                                          batch_size=self.config['batch_size'],
                                          num_workers=self.config['num_workers'],
                                          shuffle=self.config['shuffle'],
                                          pin_memory=self.config['pin_memory'],
                                          drop_last=self.config['drop_last']
                                          )
