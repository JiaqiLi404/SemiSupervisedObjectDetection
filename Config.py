# @Time : 2023/2/23 15:43 
# @Author : Li Jiaqi
# @Description :
from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
import platform

LoggingConfig = dict(
    path="log.txt"
)
DataLoaderConfig = dict(
    dataset="LoveDA",
    LoveDA=dict(
        image_dir=dict(
            train=["E:\\Research\\Datas\\AreialImage\\2021LoveDA\\Train\\Rural\\images_png",
                   "/root/autodl-tmp/LoveDA/Train/Rural/images_png"],
            valid=["E:\\Research\\Datas\\AreialImage\\2021LoveDA\\Val\\Rural\\images_png",
                   "/root/autodl-tmp/LoveDA/Val/Rural/images_png"],
        ),
        mask_dir=dict(
            train=["E:\\Research\\Datas\\AreialImage\\2021LoveDA\\Train\\Rural\\masks_png",
                   "/root/autodl-tmp/LoveDA/Train/Rural/masks_png"],
            valid=["E:\\Research\\Datas\\AreialImage\\2021LoveDA\\Val\\Rural\\masks_png",
                   "/root/autodl-tmp/LoveDA/Val/Rural/masks_png"],
        ),
        transforms=Compose([
            RandomCrop(512, 512),
            OneOf([
                HorizontalFlip(True),
                VerticalFlip(True),
                RandomRotate90(True)
            ], p=0.75),
            # Normalize(mean=(123.675, 116.28, 103.53),
            #           std=(58.395, 57.12, 57.375),
            #           max_pixel_value=1, always_apply=True),
        ]),
        batch_size=3 if platform.system().lower() == 'windows' else 9,
        num_workers=8,
        drop_last=True,  # whether abandon the samples out of batch
        shuffle=True,  # whether to choose the samples in random order
        pin_memory=True  # whether to keep the data in pin memory
    )
)
