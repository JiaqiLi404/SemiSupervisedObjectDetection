import os

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

import models.Loss as myLoss


class SegFormerModel(nn.Module):
    def __init__(self, pretrain_weight=None, lr=None, weight_decay=None, scheduler=None, device="cuda:0",
                 use_dice_loss=True, num_labels=1, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True,
                                                                      num_labels=num_labels,
                                                                      reshape_last_stage=True)
        self.model.config.output_hidden_states = True
        self.model.to(device)
        self.device = device

        if pretrain_weight:
            self.load_state_dict(
                torch.load(os.path.join('checkpoints', pretrain_weight), map_location=torch.device(device)))
            print("Pretrained model loaded")

        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, self.model.parameters()),
                                          lr=lr,  # config.ModelConfig['lr'],
                                          weight_decay=weight_decay,  # config.ModelConfig['weight_decay'],
                                          betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=scheduler)  # config.ModelConfig['scheduler']
        # loss_function = torch.nn.L1Loss()
        self.loss_function = myLoss.SegmentationLoss(1, loss_type='dice', activation='none')
        self.mse_loss_function = myLoss.SegmentationLoss(1, loss_type='mse')
        self.activation_fn = torch.nn.Sigmoid()
        self.use_dice_loss = use_dice_loss
        self.num_labels = num_labels

    def frozen_encoder(self, layers_num=None):
        if layers_num is None:
            layers_num = self.model.config.num_encoder_blocks
        for param in self.model.segformer.encoder.block[:layers_num].parameters():
            param.requires_grad = False

    def unfroze_encoder(self):
        for param in self.model.segformer.encoder.block.parameters():
            param.requires_grad = True

    def predict(self, img, mask=None, isEval=True):
        if not isEval:
            self.model.eval()
        img = img.to(self.device)
        if mask is not None:
            mask = mask.to(self.device, dtype=torch.int64)
        outputs = self.model(pixel_values=img,
                             labels=mask)  # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = outputs.logits
        size = list(img.shape)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=size[2:],  # (height, width)
                                                     mode='bilinear',
                                                     align_corners=False)

        # Second, apply argmax on the class dimension
        # upsampled_logits = upsampled_logits.argmax(dim=1)
        predict_masks = self.activation_fn(upsampled_logits)
        predict_masks = torch.squeeze(predict_masks, 1)
        loss = outputs.loss
        if mask is None:
            return predict_masks
        if self.use_dice_loss:
            loss = self.loss_function(predict_masks, mask)
            return loss, predict_masks
        return loss, predict_masks

    def eval_one_epoch(self, imgs, masks):  # return loss, predict_mask
        self.model.eval()
        with torch.no_grad():
            return self.predict(imgs, masks)

    def train_one_epoch(self, imgs, masks):  # return loss, predict_mask
        self.model.train()
        loss, predict_masks = self.predict(imgs, masks, isEval=False)
        self.train_from_loss(loss)
        return loss, predict_masks

    def train_from_loss(self, loss):
        self.optimizer.zero_grad()
        if not torch.isnan(loss): loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.2)
        self.optimizer.step()

    def scheduler_step(self):
        self.scheduler.step()

    def show_mask(self, vis, img, mask, title=""):
        mask_img = img.cpu().numpy()
        if mask is not None:
            mask = mask.detach().cpu().numpy()
            mask_img[0, :, :] = mask
        vis.image(mask_img, opts=dict(title=title))
        # mask_img = mask_img.transpose((1, 2, 0))
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join('figures', 'show mask ' , " .png"), 255 * mask_img)

    def eval_one_epoch_without_mask(self, imgs):  # return loss, predict_mask
        self.model.eval()
        with torch.no_grad():
            imgs = imgs.to(self.device)
            outputs = self.model(pixel_values=imgs)  # logits are of shape (batch_size, num_labels, height/4, width/4)
            logits = outputs.logits
            size = list(imgs.shape)

            # First, rescale logits to original image size
            upsampled_logits = nn.functional.interpolate(logits,
                                                         size=size[2:],  # (height, width)
                                                         mode='bilinear',
                                                         align_corners=False)

            upsampled_logits = self.activation_fn(upsampled_logits)

            loss = self.mse_loss_function(imgs, upsampled_logits)

            return loss, upsampled_logits

    def train_one_epoch_without_mask(self, imgs):
        self.model.train()
        # cuda tensor
        imgs = imgs.to(self.device)
        outputs = self.model(pixel_values=imgs)

        # logits = outputs.logits.cpu()
        logits = outputs.logits
        size = list(imgs.shape)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=size[2:],  # (height, width)
                                                     mode='bilinear',
                                                     align_corners=False)

        upsampled_logits = self.activation_fn(upsampled_logits)

        loss = self.mse_loss_function(imgs, upsampled_logits)
        self.train_from_loss(loss)

        return loss, upsampled_logits
