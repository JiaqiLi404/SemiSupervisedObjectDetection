# @Time : 2023/6/20 18:00
# @Author : Li Jiaqi
# @Description :
import os
from models.ViT_Decoder import Decoder
from models.ViT_EncoderDecoder import EncoderDecoder
import models.Loss as myLoss
import torch.nn as nn
import torch
import config
import numpy as np
import cv2
import visdom

class VitSegModel(nn.Module):
    def __init__(self, pretrain_weight='vit-seg-without-autoencoder epoch 38 train 0.230 eval 0.274 fps 0.70.pth',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.encoder_model.cuda()
        self.decoder = Decoder(img_size=(config.ModelConfig['imgh'], config.ModelConfig['imgw']), out_chans=1,
                               patch_size=self.encoder_model.patch_size, 
                               depth=self.encoder_model.n_blocks,
                               embed_dim=self.encoder_model.embed_dim, 
                               num_heads=self.encoder_model.num_heads).cuda()
        self.model = EncoderDecoder(self.encoder_model, self.decoder)

        # load the pretrained weights
        self.model.load_state_dict(torch.load(os.path.join('checkpoints', pretrain_weight)))
        print('pretrained model loaded')

        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, self.model.parameter()),
                                          lr=config.ModelConfig['lr'], 
                                          weight_decay=config.ModelConfig['weight_decay'],
                                          betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.ModelConfig['scheduler'])
        # loss_function = torch.nn.L1Loss()
        self.loss_function = myLoss.SegmentationLoss(1, loss_type='dice', activation='none')
        self.activation_fn = torch.nn.Sigmoid()

    def predict(self, batch_images):
        with torch.no_grad():
            self.model.eval()
            batch_images = batch_images.cuda()
            output = self.model(batch_images)
            mask = self.activation_fn(output)
        return mask

    def train_one_epoch(self, imgs, masks):
        self.model.train()
        # cuda tensor
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = masks.unsqueeze(dim=1)

        # calculate loss
        predict_masks = self.predict(imgs)
        loss = self.loss_function(predict_masks, masks)

        # train & decrease loss
        self.train_from_loss(loss)

    def train_from_loss(self, loss):
        self.optimizer.zero_grad()
        if not torch.isnan(loss): loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value=1.2)
        self.optimizer.step()

    def scheduler_step(self):
        self.scheduler.step()

    def show_mask(self, img, mask):
        vis = visdom.Visdom(env='plot1')

        mask = mask.cpu().numpy()[0]
        mask_img = np.copy(img)
        mask_img[0, :, :] = mask
        vis.image(mask_img)
        mask_img = mask_img.transpose((1, 2, 0))
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join('figures', 'show mask ' , " .png"), 255 * mask_img)

