from itertools import product
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
from Utils import product

root_path = "../"
pretained_model = "nvidia/mit-b5"
visdom_display_freq = 10  # send image to visdom every 5 epoch


# python -m visdom.server

def Prediction(weight,eval_dataLoader):
    model = SegFormerModel(pretrain_weight=weight,lr=best_hyperparameters['lr'],
                           weight_decay=best_hyperparameters['weight_decay'],
                           scheduler=best_hyperparameters['scheduler'], num_labels=3)
    for img, _, _, _ in eval_dataLoader:
        # forward
        loss, recovery = model.eval_one_epoch_without_mask(imgs=img)

        for img_i in range(img.shape[0]):
            model.show_mask(vis_eval, img[img_i], None, title="Ground Truth")
            model.show_mask(vis_eval, recovery[img_i].detach(), None,
                            title="Recovered Image")


def Train(model, train_dataloader, eval_dataLoader, train_unlabel_dataloader, eval_unlabel_dataloader,
          epoch_num=config.ModelConfig['epoch_num'],
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
        for img, _, _, _ in train_dataloader:
            # forward
            loss, recovery = model.train_one_epoch_without_mask(imgs=img)
            train_epoch_loss.append(loss.item())

            if len(train_epoch_loss) % visdom_display_freq == 0:
                model.show_mask(vis_train, img[0], None, title="Ground Truth")
                model.show_mask(vis_train, recovery[0].detach(), None,
                                title="Recovered Image epoch {0}".format(epoch_i))

        if train_unlabel_dataloader:
            for img, _, _, _ in train_unlabel_dataloader:
                # forward
                loss, recovery = model.train_one_epoch_without_mask(imgs=img)
                train_epoch_loss.append(loss.item())

                if len(train_epoch_loss) % visdom_display_freq == 0:
                    model.show_mask(vis_train, img[0], None, title="Ground Truth")
                    model.show_mask(vis_train, recovery[0].detach(), None,
                                    title="Recovered Image epoch {0}".format(epoch_i))

        train_loss = sum(train_epoch_loss) / (
                len(train_dataloader) + (len(train_unlabel_dataloader) if train_unlabel_dataloader else 0))
        train_loss_path.append(train_loss)
        model.scheduler_step()

        # evaluation
        s_time = time.time()
        model.eval()
        eval_epoch_loss = []
        with torch.no_grad():
            for img, _, _, _ in eval_dataLoader:
                # forward
                loss, recovery = model.eval_one_epoch_without_mask(imgs=img)
                eval_epoch_loss.append(loss.item())

                if len(eval_epoch_loss) % visdom_display_freq == 0:
                    model.show_mask(vis_eval, img[0], None, title="Ground Truth")
                    model.show_mask(vis_eval, recovery[0].detach(), None,
                                    title="Recovered Image epoch {0}".format(epoch_i))
            if eval_unlabel_dataloader:
                for img, _, _, _ in eval_unlabel_dataloader:
                    # forward
                    loss, recovery = model.eval_one_epoch_without_mask(imgs=img)
                    eval_epoch_loss.append(loss.item())

                    if len(eval_epoch_loss) % visdom_display_freq == 0:
                        model.show_mask(vis_eval, img[0], None, title="Ground Truth")
                        model.show_mask(vis_eval, recovery[0].detach(), None,
                                        title="Recovered Image epoch {0}".format(epoch_i))
        eval_loss = sum(eval_epoch_loss) / (
                len(eval_dataLoader) + (len(eval_unlabel_dataloader) if eval_unlabel_dataloader else 0))
        eval_loss_path.append(eval_loss)
        fps = (time.time() - s_time) / len(eval_dataLoader)

        print(
            'epoch {0} train_loss: {1:.6f} eval_loss: {2:.6f} fps {3:.2f}'.format(epoch_i, train_loss, eval_loss, fps))

        if train_loss + eval_loss < best_loss:
            best_loss = train_loss + eval_loss
            best_epoch = epoch_i
            if save_model:
                torch.save(model.state_dict(),
                           os.path.join('{0}/checkpoints'.format(root_path),
                                        'segFormer_autoencoder_epoch_{0}_train_{1:.3f}_eval_{2:.3f}_fps_{3:.2f}.pth'
                                        .format(epoch_i, train_loss, eval_loss, fps)))


    if loss_plot:
        print('**********FINISH**********')
        plt.title(loss_plot)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim((0, 100))
        plt.plot(range(len(train_loss_path)), train_loss_path, color='blue', label='train')
        plt.plot(range(len(eval_loss_path)), eval_loss_path, color='yellow', label='eval')
        plt.legend()
        plt.savefig(os.path.join('{0}/figures'.format(root_path), "_".join(loss_plot.split(" ")) + ".png"))
        plt.show()

    return best_loss, best_epoch


def Hyperparameter_Tuning(lr, weight_decay, scheduler, epochs=30):
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

    hyperparameter_sets = product(lr, weight_decay, scheduler, shuffle=True)
    best_loss = 100
    best_hyperparameters = {
        "lr": None,
        "weight_decay": None,
        "scheduler": None
    }
    for (_lr, _weight_decay, _scheduler) in hyperparameter_sets[:9]:
        print("Training model (hyperparameter tunning) for lr={0}, weight_decay={1}, scheduler={2}"
              .format(_lr, _weight_decay, _scheduler))
        model = SegFormerModel(lr=_lr, weight_decay=_weight_decay, scheduler=_scheduler, num_labels=3)
        loss, _ = Train(model, train_dataloader, validation_dataloader, None, epoch_num=epochs, save_model=False)
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
    vis_train = visdom.Visdom(env="SegFormerAutoencoder_Train")
    vis_eval = visdom.Visdom(env="SegFormerAutoencoder_Evaluation")
    vis_pred = visdom.Visdom(env="SegFormerAutoencoder_Prediction")

    # set hyperparameter list
    best_hyperparameters = {
        "lr": 2e-5,
        "weight_decay": 5e-5,
        "scheduler": 0.97
    }
    # best_hyperparameters = Hyperparameter_Tuning(lr=[1e-4,7e-5,5e-5,3e-5,1e-5,5e-6], weight_decay=[5e-5], scheduler=[0.97])

    label_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="train")
    # unlabel_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="unlabeled")
    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")

    unlabel_dataset = archaeological_georgia_biostyle_dataloader.SitesBingBook(config.DataLoaderConfig["unlabeledset"],
                                                                               None,
                                                                               config.DataLoaderConfig["transforms"],
                                                                               has_mask=False)
    train_unlabel_data_num = math.floor(len(unlabel_dataset) * 0.8)
    train_dataset, validation_dataset = torch.utils.data.random_split(unlabel_dataset, [train_unlabel_data_num,
                                                                                        len(unlabel_dataset) - train_unlabel_data_num])
    train_unlabel_dataloader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                                      dataset=train_dataset,
                                                                                      flag="unlabeled")
    validation_unlabel_dataloader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                                           dataset=validation_dataset,
                                                                                           flag="unlabeled")

    print('Labeled data batch amount: {0}, evaluation data batch amount: {1}'.format(len(label_dataLoader),
                                                                                     len(eval_dataLoader)))

    print('In unlabeled samples, training batch amount: {0}, evaluation batch amount: {1}'.format(
        len(train_unlabel_dataloader),
        len(validation_unlabel_dataloader)))

    print("Training model for lr={0}, weight_decay={1}, scheduler={2}".format(best_hyperparameters['lr'],
                                                                              best_hyperparameters['weight_decay'],
                                                                              best_hyperparameters['scheduler']))

    # train with dice loss
    model = SegFormerModel(lr=best_hyperparameters['lr'],
                           weight_decay=best_hyperparameters['weight_decay'],
                           scheduler=best_hyperparameters['scheduler'], num_labels=3)
    # Train(model, label_dataLoader, eval_dataLoader, train_unlabel_dataloader, validation_unlabel_dataloader,
    #       save_model=True,
    #       loss_plot="Loss Performance of SegFormer Autoencoder", epoch_num=50)
    Prediction('segFormer_autoencoder_epoch_28_train_19.970_eval_17.657_fps_3.10.pth',eval_dataLoader)
