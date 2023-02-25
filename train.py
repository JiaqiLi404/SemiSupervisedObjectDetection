import dataLoader
import Config
import logging
import utils.Register as Register
import model.loss as myLoss

import torch
import segmentation_models_pytorch as smp


def evaluate():
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        for img, mask in validDataLoader:
            img = img.to(device="cuda:0", dtype=torch.float32)
            img = img.transpose(3, 1).contiguous()
            real_mask = mask['ind_mask'].to(device="cuda:0", dtype=torch.float32)
            predict_mask = model(img)
            # print(predict_mask)
            loss = loss_fun(predict_mask, real_mask)
            valid_loss += loss.item()
        print("valid_loss:", valid_loss / len(validDataLoader))
        print("lr:", optimizer.param_groups[0]['lr'])
    # torch.cuda.empty_cache()
    return valid_loss / len(validDataLoader)


def save_checkpoint(epoch):
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch}
    path_checkpoint = "./checkpoint_{}_epoch_{}_loss.pkl".format(epoch, best_loss)
    torch.save(checkpoint, path_checkpoint)


if __name__ == '__main__':
    logging.basicConfig(filename=Config.LoggingConfig['path'])
    dataset_name = Config.DataLoaderConfig['dataset']
    trainDataLoader = Register.DataLoaderRegister[dataset_name + 'Loader'](Config.DataLoaderConfig[dataset_name])
    validDataLoader = Register.DataLoaderRegister[dataset_name + 'Loader'](Config.DataLoaderConfig[dataset_name],
                                                                           "valid")
    # ENCODER = 'se_resnext50_32x4d'
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, activation=ACTIVATION, classes=7,
                     in_channels=3)
    # model.half()
    model.cuda()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    loss_fun = myLoss.SegmentationLoss(7, loss_type='dice')

    # optimizer = torch.optim.SGD(params=model.parameters(), momentum=0.9, weight_decay=0.0001, lr=0.01)
    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=0.001, lr=0.0001)
    max_step = 5000
    lambda_func = lambda step: (1 - step / max_step) ** 0.9 if step < max_step else 1e-3
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    grad_batch = 1
    best_loss = 1

    for i in range(0, 100):
        print('\nEpoch: {}'.format(i))
        eval_loss = evaluate()
        if eval_loss < best_loss:
            best_loss = eval_loss
            save_checkpoint(i)
        current_batch = 0
        model.train()
        optimizer.zero_grad()
        for img, mask in trainDataLoader:
            with torch.no_grad():
                img = img.to(device="cuda:0", dtype=torch.float32)
                img = img.transpose(3, 1).contiguous()
                real_mask = mask['ind_mask'].to(device="cuda:0", dtype=torch.float32)
            predict_mask = model(img)
            # predict_mask = predict_mask.softmax(dim=1)
            # print(real_mask.shape, predict_mask.shape)
            loss = loss_fun(predict_mask, real_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=35, norm_type=2)
            current_batch += 1
            if current_batch % 10 == 0:
                print(loss.item())
            if current_batch % grad_batch == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
