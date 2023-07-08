import os.path
import sys

sys.path.append('../')
import archaeological_georgia_biostyle_dataloader
import torch
import config
import visdom
import matplotlib.pyplot as plt
import time
from models.SegFormerModel import SegFormerModel
import math

root_path = "../"
pretained_model = "nvidia/mit-b5"
visdom_display_freq = 5  # send image to visdom every 5 epoch


# python -m visdom.server

def Prediction():
    unlabel_dataLoader = archaeological_georgia_biostyle_dataloader \
        .SitesLoader(config.DataLoaderConfig, flag="unlabeled")
    model = SegFormerModel()  # with pre-trained weight
    model.eval()
    with torch.no_gard():
        dataPatches = 0
        for img, _, _, _ in unlabel_dataLoader:
            predict_mask = model.predict(
                img=img)  # logits are of shape (batch_size, num_labels, height/4, width/4)
            dataPatches += 1
            if dataPatches % visdom_display_freq == 0:
                model.show_mask(vis_pred, img[0], None, title="Raw Image {0}".format(dataPatches))
                model.show_mask(vis_pred, img[0], predict_mask[0], title="Predicted Mask epoch{0}".format(dataPatches))


def Train(model, train_dataloader, eval_dataLoader, epoch_num=config.ModelConfig['epoch_num'],
          save_model=False, loss_plot=None):
    train_loss_path = []
    eval_loss_path = []
    best_loss = 100
    best_epoch = -1
    # metric = load_metric("mean_iou")
    for epoch_i in range(epoch_num):
        # train
        model.train()
        train_epoch_loss = []
        for img, mask, _, _ in train_dataloader:
            # forward
            loss, predict_mask = model.train_one_epoch(imgs=img, masks=mask)
            train_epoch_loss.append(loss.item())

            if len(train_epoch_loss) % visdom_display_freq == 0:
                model.show_mask(vis_train, img[0], mask[0], title="Ground Truth")
                model.show_mask(vis_train, img[0], predict_mask[0], title="Predicted Mask epoch {0}".format(epoch_i))

        train_loss = sum(train_epoch_loss) / len(train_dataloader)
        train_loss_path.append(train_loss)
        model.scheduler_step()

        # evaluation
        s_time = time.time()
        model.eval()
        eval_epoch_loss = []
        with torch.no_grad():
            for img, mask, _, _ in eval_dataLoader:
                # forward
                loss, predict_mask = model.eval_one_epoch(imgs=img, masks=mask)
                eval_epoch_loss.append(loss.item())

                if len(eval_epoch_loss) % visdom_display_freq == 0:
                    model.show_mask(vis_eval, img[0], mask[0], title="Ground Truth")
                    model.show_mask(vis_eval, img[0], predict_mask[0], title="Predicted Mask epoch {0}".format(epoch_i))
        eval_loss = sum(eval_epoch_loss) / len(eval_dataLoader)
        eval_loss_path.append(eval_loss)
        fps = len(eval_dataLoader) / (time.time() - s_time)

        print(
            'epoch {0} train_loss: {1:.6f} eval_loss: {2:.6f} fps {3:.2f}'.format(epoch_i, train_loss, eval_loss, fps))

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch_i
            if save_model:
                torch.save(model.state_dict(),
                           os.path.join('{0}/checkpoints'.format(root_path),
                                        'segFormer_epoch_{0}_train_{1:.3f}_eval_{2:.3f}_fps_{3:.2f}.pth'
                                        .format(epoch_i, train_loss, best_loss, fps)))

    if loss_plot:
        print('**********FINISH**********')
        plt.title(loss_plot)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim((0, 1))
        plt.plot(range(config.ModelConfig['epoch_num']), train_loss_path, color='blue', label='train')
        plt.plot(range(config.ModelConfig['epoch_num']), eval_loss_path, color='yellow', label='eval')
        plt.legend()
        plt.savefig(os.path.join('{0}/figures'.format(root_path), "_".join(loss_plot.split(" ")) + ".png"))
        plt.show()

    return best_loss, best_epoch


def Hyperparameter_Tuning(lr, weight_decay, scheduler, epochs=10):
    label_dataset = archaeological_georgia_biostyle_dataloader.SitesBingBook(config.DataLoaderConfig["dataset"],
                                                                             config.DataLoaderConfig["maskdir"],
                                                                             config.DataLoaderConfig["transforms"])
    train_data_num = math.floor(len(label_dataset) * 0.8)
    train_dataset, validation_dataset = torch.utils.data.random_split(label_dataset, [train_data_num,
                                                                                      len(label_dataset) - train_data_num])
    train_dataloader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                              dataset=train_dataset, flag="train")
    validation_dataloader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                                   dataset=validation_dataset,
                                                                                   flag="train")

    print('Training data batch amount: {0}, Validation data batch amount: {1}'.format(len(train_dataloader),
                                                                                      len(validation_dataloader)))

    best_loss = 100
    best_hyperparameters = {
        "lr": None,
        "weight_decay": None,
        "scheduler": None
    }
    for _lr in lr:
        for _weight_decay in weight_decay:
            for _scheduler in scheduler:
                print("Training model (hyperparameter tunning) for lr={0}, weight_decay={1}, scheduler={2}"
                      .format(_lr, _weight_decay, _scheduler))
                model = SegFormerModel(
                    pretrain_weight='segFormer_autoencoder_epoch_18_train_22.885_eval_15.023_fps_3.37.pth',
                    lr=_lr, weight_decay=_weight_decay, scheduler=_scheduler)
                model.frozen_encoder()
                loss, trained_epoch = Train(model, train_dataloader, validation_dataloader,
                                            epoch_num=epochs, save_model=False)
                print(
                    "    Model loss (hyperparameter tunning) for lr={0}, weight_decay={1}, scheduler={2}: {3:.4f}".format(
                        _lr, _weight_decay, _scheduler, loss))
                if loss < best_loss:
                    best_loss = loss
                    best_hyperparameters = {
                        "lr": _lr,
                        "weight_decay": _weight_decay,
                        "scheduler": _scheduler
                    }

    return best_hyperparameters


if __name__ == '__main__':
    device = "cuda:0"
    vis_train = visdom.Visdom(env="SegFormer_Train")
    vis_eval = visdom.Visdom(env="SegFormer_Evaluation")
    vis_pred = visdom.Visdom(env="SegFormer_Prediction")

    # set hyperparameter list
    best_hyperparameters = {
        "lr": 2e-5,
        "weight_decay": 5e-5,
        "scheduler": 0.97
    }
    # best_hyperparameters = Hyperparameter_Tuning(lr=[30e-5, 15e-5, 10e-6, 5e-6, 1e-6], weight_decay=[5e-5],
    #                                              scheduler=[0.97])

    label_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="train")
    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    print('Labeled data batch amount: {0}, evaluation data batch amount: {1}'.format(len(label_dataLoader),
                                                                                     len(eval_dataLoader)))

    print("Training model for lr={0}, weight_decay={1}, scheduler={2}".format(best_hyperparameters['lr'],
                                                                              best_hyperparameters['weight_decay'],
                                                                              best_hyperparameters['scheduler']))

    model = SegFormerModel(pretrain_weight=None,
                           lr=best_hyperparameters['lr'], weight_decay=best_hyperparameters['weight_decay'],
                           scheduler=best_hyperparameters['scheduler'])
    model.frozen_encoder(layers=[1, 2, 3])
    model.add_prompt_token([5, 5, 5, 0])
    Train(model, label_dataLoader, eval_dataLoader, save_model=True,
          loss_plot="Loss Performance of SegFormer")
