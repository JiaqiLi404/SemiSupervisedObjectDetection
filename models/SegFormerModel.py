import os
import models.Loss as myLoss
import torch.nn as nn
import torch
from transformers import SegformerForSemanticSegmentation, SegformerModel


class SegFormerModel(nn.Module):
    def __init__(self, pretrain_weight=None, lr=None, weight_decay=None, scheduler=None, device="cuda:0", *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        id2label = {0: 'background', 1: 'archaeological'}
        label2id = {'background': 0, 'archaeological': 1}
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True,
                                                                      num_labels=len(id2label), id2label=id2label,
                                                                      label2id=label2id,
                                                                      reshape_last_stage=True)
        self.model.config.output_hidden_states = True
        self.model.to(device)
        self.device = device

        if pretrain_weight:
            self.model.load_state_dict(
                torch.load(os.path.join('checkpoints', pretrain_weight), map_location=torch.device(device)))
            print("Pretrained model loaded")

        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, self.model.parameters()),
                                          lr=lr,  # config.ModelConfig['lr'],
                                          weight_decay=weight_decay,  # config.ModelConfig['weight_decay'],
                                          betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=scheduler)  # config.ModelConfig['scheduler']
        # loss_function = torch.nn.L1Loss()
        # self.loss_function = myLoss.SegmentationLoss(1, loss_type='dice', activation='none')
        self.activation_fn = torch.nn.Sigmoid()

    def predict(self, img):
        self.model.eval()
        img = img.to(self.device)
        outputs = self.model(pixel_values=img)  # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = outputs.logits.cpu()
        size = list(img.shape)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=size[2:],  # (height, width)
                                                     mode='bilinear',
                                                     align_corners=False)

        # Second, apply argmax on the class dimension
        predict_mask = upsampled_logits.argmax(dim=1)

        return predict_mask

    def loss_function(self,pred,gt):
        pred = pred.to(self.device)
        gt = gt.to(self.device, dtype=torch.int64)
        outputs = self.model(pixel_values=pred)  # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = outputs.logits.cpu()
        size = list(pred.shape)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=size[2:],  # (height, width)
                                                     mode='bilinear',
                                                     align_corners=False)

        valid_mask = (gt >= 0).float()
        loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss_fct(upsampled_logits.squeeze(1), gt.float())
        loss = (loss * valid_mask).mean()
        return loss

    def eval_one_epoch(self, imgs, masks):
        self.model.eval()
        with torch.no_grad():
            imgs = imgs.to(self.device)
            masks = masks.to(self.device, dtype=torch.int64)
            outputs = self.model(pixel_values=imgs,
                                 labels=masks)  # logits are of shape (batch_size, num_labels, height/4, width/4)
            logits = outputs.logits.cpu()
            size = list(imgs.shape)

            # First, rescale logits to original image size
            upsampled_logits = nn.functional.interpolate(logits,
                                                         size=size[2:],  # (height, width)
                                                         mode='bilinear',
                                                         align_corners=False)

            # Second, apply argmax on the class dimension
            predict_masks = upsampled_logits.argmax(dim=1)

            return outputs.loss, predict_masks

    def train_one_epoch(self, imgs, masks):
        self.model.train()
        # cuda tensor
        imgs = imgs.to(self.device)
        masks = masks.to(self.device,dtype=torch.int64)
        outputs = self.model(pixel_values=imgs, labels=masks)

        # logits = outputs.logits.cpu()
        logits = outputs.logits
        size = list(imgs.shape)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=size[2:],  # (height, width)
                                                     mode='bilinear',
                                                     align_corners=False)

        # Second, apply argmax on the class dimension
        predict_masks = upsampled_logits.argmax(dim=1)

        # _loss = self.loss_function(predict_masks, masks)
        self.train_from_loss(outputs.loss)
        return outputs.loss, predict_masks

    def train_from_loss(self, loss):
        self.optimizer.zero_grad()
        if not torch.isnan(loss): loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.2)
        self.optimizer.step()

    def scheduler_step(self):
        self.scheduler.step()

    def show_mask(self, vis, img, mask, title=""):
        mask_img = img.cpu().numpy()
        mask= mask.cpu().numpy()
        if mask is not None:
            mask_img[0, :, :] = mask
        vis.image(mask_img, opts=dict(title=title))
        # mask_img = mask_img.transpose((1, 2, 0))
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join('figures', 'show mask ' , " .png"), 255 * mask_img)
