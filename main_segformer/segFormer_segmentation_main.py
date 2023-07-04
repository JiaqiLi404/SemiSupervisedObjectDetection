# @Time : 2023/4/22 18:00
# @Author : Li Jiaqi
# @Description :
import os.path

import archaeological_georgia_biostyle_dataloader
import torch
import config
import visdom
import numpy as np
from models.ViT_Decoder import Decoder
from models.ViT_EncoderDecoder import EncoderDecoder
import cv2
import models.Loss as myLoss
import matplotlib.pyplot as plt
import time
import sys


# python -m visdom.server
def train():
    loss_path_train = []
    loss_path_eval = []
    best_loss = 100
    for epoch_i in range(config.ModelConfig['epoch_num']):
        epoch_loss = []
        for img, mask, _, _ in label_dataLoader:
            predict_mask, epoch_loss = train_step(img, mask, epoch_loss)
            # show the image to Visdom
            if len(epoch_loss) % 20 == 0:
                bing_image_numpy = img.numpy()
                bing_image_numpy = bing_image_numpy[0]
                vis.image(bing_image_numpy)
                predict_mask_numpy = predict_mask.numpy()
                predict_mask_numpy = predict_mask_numpy[0]
                vis.image(predict_mask_numpy)
        # # save model
        # if epoch_i % 5 == 0:
        #     torch.save(ae_model.state_dict(),
        #                os.path.join('checkpoints',
        #                             'vit-seg-epoch-{0}-loss-{1:.3f}.pth'.format(epoch_i, sum(epoch_loss))))
        scheduler.step()
        train_loss = sum(epoch_loss) / len(label_dataLoader)

        s_time = time.time()
        eval_loss = eval(ae_model, eval_dataLoader)
        fps = len(eval_dataLoader) / (time.time() - s_time)

        # save the best model
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(ae_model.state_dict(),
                       os.path.join('checkpoints', 'vit-seg epoch {0} train {1:.3f} eval {2:.3f} fps {3:.2f}.pth'
                                    .format(epoch_i, train_loss, best_loss, fps)))

        print(
            'epoch {0} train_loss: {1:.6f} eval_loss: {2:.6f} fps {3:.2f}'.format(epoch_i, train_loss, eval_loss, fps))
        loss_path_train.append(train_loss)
        loss_path_eval.append(eval_loss)

    print('**********FINISH**********')
    plt.title('Loss Performance of common ViT')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim((0, 1))
    plt.plot(range(config.ModelConfig['epoch_num']), loss_path_train, color='blue', label='train')
    plt.plot(range(config.ModelConfig['epoch_num']), loss_path_eval, color='yellow', label='eval')
    plt.legend()
    plt.savefig(os.path.join('figures', 'Loss Performance of common ViT.png'))
    plt.show()


def train_step(bing_image, bing_mask, epoch_loss):
    ae_model.train()
    # cuda tensors
    bing_image_cuda = bing_image.cuda()
    bing_mask_cuda = bing_mask.cuda()
    bing_mask_cuda = bing_mask_cuda.unsqueeze(dim=1)

    # train the AutoEncoder(decoder)
    predict_mask_cuda, bing_feature_cuda = ae_model(bing_image_cuda)
    activation_fn = torch.nn.Sigmoid()
    predict_mask_cuda = activation_fn(predict_mask_cuda)
    predict_mask = predict_mask_cuda.detach().cpu()
    resolution_loss = ae_model.patch_loss(predict_mask_cuda, bing_mask_cuda)

    loss = loss_function(predict_mask_cuda, bing_mask_cuda)
    # loss = resolution_loss
    optimizer.zero_grad()
    if not torch.isnan(loss): loss.backward()
    if len(epoch_loss) % 5 == 0:
        print('loss:{0:.6f} '.format(float(loss)))
    epoch_loss.append(float(loss))
    # torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=20, norm_type=2)
    torch.nn.utils.clip_grad_value_(ae_model.parameters(), clip_value=1.2)
    optimizer.step()
    return predict_mask, epoch_loss


def eval(model, eval_dataLoader):
    with torch.no_grad():
        model.eval()
        valid_loss = []
        for img, mask, _, _ in eval_dataLoader:
            img = img.to(device="cuda:0", dtype=torch.float32)
            real_mask = mask.to(device="cuda:0", dtype=torch.float32)
            real_mask = real_mask.unsqueeze(1)
            predict_mask, _ = model(img)
            activation_fn = torch.nn.Sigmoid()
            predict_mask_cuda = activation_fn(predict_mask)
            loss = loss_function(predict_mask_cuda, real_mask)
            valid_loss.append(float(loss.item()))

            # show the image to Visdom
            if len(valid_loss) % 2 == 0:
                img = img.cpu()
                img = img.numpy()
                img = img[0]
                real_mask = real_mask.cpu()
                real_mask_numpy = real_mask.numpy()
                real_mask_numpy = real_mask_numpy[0]
                real_mask_img = np.copy(img)
                real_mask_img[0, :, :] = real_mask_numpy
                vis.image(real_mask_img)
                predict_mask = predict_mask_cuda.cpu()
                predict_mask_numpy = predict_mask.numpy()
                predict_mask_numpy = predict_mask_numpy[0]
                predict_mask_img = np.copy(img)
                predict_mask_img[0, :, :] = predict_mask_numpy
                vis.image(predict_mask_img)
    return sum(valid_loss) / len(eval_dataLoader)


def predict(weight_file):
    dino_encoder_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dino_encoder_model.cuda()
    decoder = Decoder(img_size=(config.ModelConfig['imgh'], config.ModelConfig['imgw']),
                      patch_size=dino_encoder_model.patch_size, depth=dino_encoder_model.n_blocks,
                      embed_dim=dino_encoder_model.embed_dim, num_heads=dino_encoder_model.num_heads,
                      out_chans=1).cuda()
    ae_model = EncoderDecoder(dino_encoder_model, decoder)
    ae_model.load_state_dict(torch.load(os.path.join('checkpoints', weight_file)))
    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    i = 0
    with torch.no_grad():
        ae_model.eval()
        for img, mask, _, _ in eval_dataLoader:
            i += 1
            img = img.cuda()
            predict_mask, _ = ae_model(img)
            activation_fn = torch.nn.Sigmoid()
            predict_mask = activation_fn(predict_mask)

            img = img.cpu()
            img = img.numpy()
            img = img[0]
            real_mask = mask.cpu()
            real_mask_numpy = real_mask.numpy()
            real_mask_numpy = real_mask_numpy[0]
            real_mask_img = np.copy(img)
            real_mask_img[0, :, :] = real_mask_numpy
            vis.image(real_mask_img)
            real_mask_img = real_mask_img.transpose((1, 2, 0))
            real_mask_img = cv2.cvtColor(real_mask_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('figures', 'sample_' + str(i) + "_gt.png"), 255 * real_mask_img)
            predict_mask = predict_mask.cpu()
            predict_mask_numpy = predict_mask.numpy()
            predict_mask_numpy = predict_mask_numpy[0]
            predict_mask_img = np.copy(img)
            predict_mask_img[0, :, :] = predict_mask_numpy
            vis.image(predict_mask_img)
            predict_mask_img = predict_mask_img.transpose((1, 2, 0))
            predict_mask_img = cv2.cvtColor(predict_mask_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('figures', 'sample_' + str(i) + "_predict.png"), 255 * predict_mask_img)


if __name__ == '__main__':
    vis = visdom.Visdom(env='plot1')
    unlabel_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                                flag="unlabeled")
    label_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="train")
    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    print('Labeled data batch amount: ', len(unlabel_dataLoader) + len(label_dataLoader))

    # dino_encoder_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dino_encoder_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dino_encoder_model.cuda()
    decoder = Decoder(img_size=(config.ModelConfig['imgh'], config.ModelConfig['imgw']), out_chans=1,
                      patch_size=dino_encoder_model.patch_size, depth=dino_encoder_model.n_blocks,
                      embed_dim=dino_encoder_model.embed_dim, num_heads=dino_encoder_model.num_heads).cuda()
    ae_model = EncoderDecoder(dino_encoder_model, decoder)
    if os.path.exists(os.path.join('checkpoints', 'vit-seg-pretrain.pth')):
        ae_model.load_state_dict(torch.load(os.path.join('checkpoints', 'vit-seg-pretrain.pth')))
        print('pretrained model loaded')
    # load autoencoder pretrained encoder weight
    elif len(sys.argv) > 1:
        encoder_weight = sys.argv[1]
        dino_encoder_model.load_state_dict(torch.load(os.path.join('checkpoints', encoder_weight)))
        ae_model.froze_encoder(2)
        print('autoencoder pretrained model loaded : ', sys.argv[1])
    else:
        print("training without pretrained weights")

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, ae_model.parameters()),
                                 lr=config.ModelConfig['lr'], weight_decay=config.ModelConfig['weight_decay'],
                                 betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.ModelConfig['scheduler'])
    # loss_function = torch.nn.L1Loss()
    loss_function = myLoss.SegmentationLoss(1, loss_type='dice', activation='none')

    train()
    # predict('vit-seg-without-autoencoder epoch 38 train 0.230 eval 0.274 fps 0.70.pth')
