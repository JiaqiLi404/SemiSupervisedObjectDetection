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

def Prediction(pretrain_weight):
    model = SegFormerModel(lr=best_hyperparameters['lr'], weight_decay=best_hyperparameters['weight_decay'],
                           scheduler=best_hyperparameters['scheduler'])
    model.frozen_encoder(layers=best_hyperparameters['frozen'])
    model.add_prompt_token([10, 10, 10, 10])
    model.load_state_dict(torch.load(os.path.join('../checkpoints', pretrain_weight),
                                                 map_location=torch.device(device)), strict=False)
    model.eval()
    for img, mask, _, _ in eval_dataLoader:
        # forward
        loss, predict_mask = model.eval_one_epoch(imgs=img, masks=mask)

        for img_i in range(img.shape[0]):
            model.show_mask(vis_eval, img[img_i], mask[img_i], title="Ground Truth")
            model.show_mask(vis_eval, img[img_i], predict_mask[img_i].detach(), title="Recovered Image")


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
        plt.plot(range(len(train_loss_path)), train_loss_path, color='blue', label='train')
        plt.plot(range(len(eval_loss_path)), eval_loss_path, color='yellow', label='eval')
        plt.legend()
        plt.savefig(os.path.join('{0}/figures'.format(root_path), 'ae_pretraining'+"_".join(loss_plot.split(" ")) + ".png"))
        plt.show()

    return best_loss, best_epoch


def Hyperparameter_Tuning(lr, weight_decay, scheduler, frozen, epochs=15):
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
        "scheduler": None,
        'frozen': None
    }
    for _lr in lr:
        for _weight_decay in weight_decay:
            for _scheduler in scheduler:
                for _frozen in frozen:
                    print(
                        "Training model (hyperparameter tunning) for lr={0}, weight_decay={1}, scheduler={2}, frozen={3}"
                        .format(_lr, _weight_decay, _scheduler, _frozen))
                    model = SegFormerModel(
                        pretrain_weight=None,
                        lr=_lr, weight_decay=_weight_decay, scheduler=_scheduler)
                    model.frozen_encoder(layers=_frozen)
                    model.add_prompt_token([10, 10, 10, 10])
                    loss, trained_epoch = Train(model, train_dataloader, validation_dataloader,
                                                epoch_num=epochs, save_model=False,
                                                loss_plot="lr-{0} weight_decay-{1} scheduler-{2} frozen-{3}"
                                                .format(_lr, _weight_decay, _scheduler, _frozen))
                    print(
                        "    Model loss (hyperparameter tunning) for lr={0}, weight_decay={1}, scheduler={2}, frozen={3}: {4:.4f}".format(
                            _lr, _weight_decay, _scheduler, _frozen, loss))
                    if loss < best_loss:
                        best_loss = loss
                        best_hyperparameters = {
                            "lr": _lr,
                            "weight_decay": _weight_decay,
                            "scheduler": _scheduler,
                            'frozen': _frozen
                        }

    return best_hyperparameters


if __name__ == '__main__':
    device = "cuda:0"
    vis_train = visdom.Visdom(env="SegFormer_Train")
    vis_eval = visdom.Visdom(env="SegFormer_Evaluation")
    vis_pred = visdom.Visdom(env="SegFormer_Prediction")

    # set hyperparameter list
    best_hyperparameters = {
        "lr": 4e-5,
        "weight_decay": 5e-5,
        "scheduler": 0.97,
        'frozen': [0,1]
    }
    # best_hyperparameters = Hyperparameter_Tuning(lr=[15e-5, 5e-5, 10e-6, 5e-6, 1e-6], weight_decay=[5e-5],
    #                                              scheduler=[0.97],
    #                                              frozen=[[0], [1], [2], [3], [0, 1], [0, 2], [1, 2], [1, 3]])

    label_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="train")
    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    print('Labeled data batch amount: {0}, evaluation data batch amount: {1}'.format(len(label_dataLoader),
                                                                                     len(eval_dataLoader)))

    print("Training model for lr={0}, weight_decay={1}, scheduler={2} frozen={3}".format(best_hyperparameters['lr'],
                                                                                         best_hyperparameters[
                                                                                             'weight_decay'],
                                                                                         best_hyperparameters[
                                                                                             'scheduler'],
                                                                                         best_hyperparameters[
                                                                                             'frozen']))

    model = SegFormerModel(pretrain_weight='few-shot seg-former epoch 174 train 142.893 eval 7.715 fps 3.67.pth',
                           lr=best_hyperparameters['lr'], weight_decay=best_hyperparameters['weight_decay'],
                           scheduler=best_hyperparameters['scheduler'])
    model.frozen_encoder(layers=best_hyperparameters['frozen'])
    model.add_prompt_token([10, 10, 10, 10])
    # Train(model, label_dataLoader, eval_dataLoader, save_model=True,
    #       loss_plot="Loss Performance of SegFormer transfer")
    Prediction('ae pretraining segFormer_epoch_48_train_0.114_eval_0.351_fps_0.65.pth')
