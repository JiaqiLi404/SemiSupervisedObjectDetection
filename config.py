# @Time : 2023/2/23 15:43 
# @Author : Li Jiaqi
# @Description :
from albumentations import Compose, OneOf, Normalize, Resize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
import platform

LoggingConfig = dict(
    path="log.txt"
)
ModelConfig = dict(
    imgh=512,
    imgw=512,
    lr=1e-5,
    weight_decay=5e-5,
    epoch_num=50,
    scheduler=0.97
)
DataLoaderConfig = dict(
    dataset="E:\\Research\\Datas\\AreialImage\\ArchaeologicalSitesDetection\\georgia_cleaned_bing_train",
    evalset="E:\\Research\\Datas\\AreialImage\\ArchaeologicalSitesDetection\\georgia_cleaned_bing_test",
    unlabeledset="E:\\Research\\Datas\\AreialImage\\ArchaeologicalSitesDetection\\new_images",
    pseudoset="E:\\Research\\Datas\\AreialImage\\ArchaeologicalSitesDetection\\pseudo",
    maskdir="E:\\Research\\Datas\\AreialImage\\ArchaeologicalSitesDetection\\masks",
    transforms=Compose([
        RandomCrop(500, 500),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(0, 0, 0),
                  std=(255, 255, 255),
                  max_pixel_value=1, always_apply=True),
        Resize(height=ModelConfig['imgh'], width=ModelConfig['imgw'])
    ]),
    batch_size=7 if platform.system().lower() == 'windows' else 20,
    num_workers=10,
    drop_last=False,  # whether abandon the samples out of batch
    shuffle=True,  # whether to choose the samples in random order
    pin_memory=True  # whether to keep the data in pin memory
)
