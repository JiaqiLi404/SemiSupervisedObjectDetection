import segmentation_models_pytorch as smp
import os.path

import archaeological_georgia_biostyle_dataloader
import torch
import config
import visdom
import matplotlib.pyplot as plt
import models.Loss as myLoss
import time
import numpy as np
import cv2


# python -m visdom.server
'''This function is used for baseline model pediction.
   The output will be two images illustrating the ground truth and the prediction 
   Input: model_name: string indicating the model, such as 'DeepLabV3+'
          weight_name: string indicating the parameter, just used to check the pretrained model
''' 
def predict(model_name, weight_name):
    model = model_ensemble[model_name]
    model.cuda()
    if os.path.exists(os.path.join('checkpoints', weight_name)):
        model.load_state_dict(torch.load(os.path.join('checkpoints', weight_name)))
        print('pretrained model loaded')
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    i = 0
    with torch.no_grad():
        model.eval()
        for img, mask, _, _ in eval_dataLoader:
            i += 1
            img = img.to(device="cuda:0", dtype=torch.float32)
            real_mask = mask.to(device="cuda:0", dtype=torch.float32)
            real_mask = real_mask.unsqueeze(1)
            predict_mask = model(img)
            activation_fn = torch.nn.Sigmoid()
            predict_mask = activation_fn(predict_mask)

            img = img.cpu()
            img = img.numpy()
            img = img[0]
            real_mask = real_mask.cpu()
            real_mask_numpy = real_mask.numpy()
            real_mask_numpy = real_mask_numpy[0]
            real_mask_img = np.copy(img)
            real_mask_img[0, :, :] = real_mask_numpy
            vis.image(real_mask_img)
            predict_mask = predict_mask.cpu()
            predict_mask_numpy = predict_mask.numpy()
            predict_mask_numpy = predict_mask_numpy[0]
            predict_mask_img = np.copy(img)
            predict_mask_img[0, :, :] = predict_mask_numpy
            vis.image(predict_mask_img)
            real_mask_img = real_mask_img.transpose((1, 2, 0))
            real_mask_img = cv2.cvtColor(real_mask_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('figures', 'sample ' + str(i) + " gt.png"), 255 * real_mask_img)
            predict_mask_img = predict_mask_img.transpose((1, 2, 0))
            predict_mask_img = cv2.cvtColor(predict_mask_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('figures', 'sample ' + str(i) + " predict.png"), 255 * predict_mask_img)

# return the average validation loss
def evaluate(model, loss_fun):
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        for img, mask, _, _ in eval_dataLoader:
            img = img.to(device="cuda:0", dtype=torch.float32)
            real_mask = mask.to(device="cuda:0", dtype=torch.float32)
            real_mask = real_mask.unsqueeze(1)
            predict_mask = model(img)
            loss = loss_fun(predict_mask, real_mask)
            valid_loss += loss.item()
    return valid_loss / len(eval_dataLoader)


def train():
    for name, model in model_ensemble.items():
        model.cuda()
        if os.path.exists(os.path.join('checkpoints', 'pretrain.pth')):
            model.load_state_dict(torch.load(os.path.join('checkpoints', 'pretrain.pth')))
            print('pretrained model loaded')
        else:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        loss_fun = myLoss.SegmentationLoss(1, loss_type='dice', activation='sigmoid')

        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.ModelConfig['lr'],
                                     weight_decay=config.ModelConfig['weight_decay'], betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.ModelConfig['scheduler'])
        best_loss = 10

        loss_path_train = []
        loss_path_eval = []
        for i in range(config.ModelConfig['epoch_num']):
            print('--------------------------------')
            current_batch = 0
            model.train()
            optimizer.zero_grad()
            epoch_loss = []
            for img, mask, _, _ in train_dataLoader:
                with torch.no_grad():
                    img = img.to(device="cuda:0", dtype=torch.float32)
                    real_mask = mask.to(device="cuda:0", dtype=torch.float32)
                    real_mask = real_mask.unsqueeze(1)
                predict_mask = model(img)
                # predict_mask = predict_mask.softmax(dim=1)
                # print(real_mask.shape, predict_mask.shape)
                loss = loss_fun(predict_mask, real_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=35, norm_type=2)
                current_batch += 1
                epoch_loss.append(float(loss))
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            train_loss = sum(epoch_loss) / len(train_dataLoader)
            s_time = time.time()
            eval_loss = evaluate(model, loss_fun)
            fps = len(eval_dataLoader) / (time.time() - s_time)

            # save the best model
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(),
                           os.path.join('checkpoints', name + ' epoch {0} train {1:.3f} eval {2:.3f} fps {3:.2f}.pth'
                                        .format(i, train_loss, best_loss, fps)))

            print('epoch {0} train_loss: {1:.6f} eval_loss: {2:.6f} fps: {3:.2f}'.format(i, train_loss, eval_loss, fps))
            loss_path_train.append(train_loss)
            loss_path_eval.append(eval_loss)

        print('**********FINISH**********')
        plt.title('Loss Performance of ' + name)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim((0, 1))
        plt.plot(range(config.ModelConfig['epoch_num']), loss_path_train, color='blue', label='train')
        plt.plot(range(config.ModelConfig['epoch_num']), loss_path_eval, color='yellow', label='eval')
        plt.legend()
        plt.savefig(os.path.join('figures', 'Loss Performance of ' + name + '.png'))
        plt.show()


if __name__ == '__main__':
    vis = visdom.Visdom(env='plot1')
    train_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="train")
    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    batch_amount = len(train_dataLoader)
    print('batch amounts: ', batch_amount)

    # ENCODER = 'resnet50'
    ENCODER = 'resnext101_32x8d'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = None  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'

    model_ensemble = {
        'PAN': smp.PAN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, activation=ACTIVATION,
                       classes=1, in_channels=3),
        'PSPNet': smp.PSPNet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, activation=ACTIVATION,
                             classes=1, in_channels=3),
        'LinkNet': smp.Linknet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, activation=ACTIVATION,
                               classes=1, in_channels=3),
        'Unet++': smp.UnetPlusPlus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, activation=ACTIVATION,
                                   classes=1, in_channels=3),
        'DeepLabV3+': smp.DeepLabV3Plus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, activation=ACTIVATION,
                                        classes=1, in_channels=3),
    }
    train()
    predict('DeepLabV3+', 'DeepLabV3+ epoch 49 train 0.151 eval 0.337 fps 1.34.pth')
