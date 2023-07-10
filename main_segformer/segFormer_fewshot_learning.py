# @Time : 2023/7/8 12:23
# @Author : Li Jiaqi
# @Description :
import os.path
import sys

sys.path.append('../')
import classified_dataloader
import archaeological_georgia_biostyle_dataloader
import torch
import config
import visdom
import matplotlib.pyplot as plt
import time
from models.SegFormerModel import SegFormerModel as SegModel
import math
import random
from Utils import product

root_path = "../"
pretained_model = "nvidia/mit-b5"
visdom_display_freq = 5  # send image to visdom every 5 epoch


# python -m visdom.server

def train(pretrain_weight, _lr, _weight_decay, _scheduler, category_dataloaders, eval_dataLoader,
          epoch_num=config.ModelConfig['epoch_num'], iteration_num=26, save_model=False, loss_plot=None):
    print('**************** Train *******************')
    print('lr: {0}'.format(_lr))
    model = SegModel(pretrain_weight, _lr, _weight_decay, _scheduler)
    model.add_cls_token()

    loss_path_train = []
    loss_path_eval = []
    loss_categories = [0 for i in range(len(category_dataloaders))]
    train_best_loss = 100
    train_best_epoch = 0

    category_dataloaders_iter = []
    for loader in category_dataloaders:
        category_dataloaders_iter.append(iter(loader))

    for epoch_i in range(epoch_num):
        epoch_loss = []
        model.train()

        for iter_i in range(iteration_num):
            # randomly pick two categories
            [category_1, category_2] = random.sample(list(range(len(category_dataloaders))), 2)
            try:
                category_1_img, category_1_mask = next(category_dataloaders_iter[category_1])
            except StopIteration:
                category_dataloaders_iter[category_1] = iter(category_dataloaders[category_1])
                category_1_img, category_1_mask = next(category_dataloaders_iter[category_1])

            try:
                category_2_img, category_2_mask = next(category_dataloaders_iter[category_2])
            except StopIteration:
                category_dataloaders_iter[category_2] = iter(category_dataloaders[category_2])
                category_2_img, category_2_mask = next(category_dataloaders_iter[category_2])

            category_1_img = category_1_img.to(device=device, dtype=torch.float32)
            category_2_img = category_2_img.to(device=device, dtype=torch.float32)
            category_1_mask = category_1_mask.to(device=device, dtype=torch.float32)
            category_2_mask = category_2_mask.to(device=device, dtype=torch.float32)

            # supervised loss
            category_1_loss, category_1_predicted = model.train_one_epoch(category_1_img, category_1_mask)
            category_2_loss, category_2_predicted = model.train_one_epoch(category_2_img, category_2_mask)
            loss_categories[category_1] = loss_categories[category_1] * 0.5 + category_1_loss * 0.5
            loss_categories[category_2] = loss_categories[category_2] * 0.5 + category_2_loss * 0.5

            # intra-loss


            epoch_loss.append(float(category_1_loss.item() + category_2_loss.item()) / 2)

            # show results
            if len(epoch_loss) % 5 == 0:
                model.show_mask(vis_train, category_1_img[0], category_1_mask[0], title="Ground Truth")
                model.show_mask(vis_train, category_1_img[0], category_1_predicted[0], "Predicted")
                print("loss:", epoch_loss[-1])

        train_loss = sum(epoch_loss) / iteration_num
        loss_path_train.append(train_loss)
        model.scheduler_step()

        # eval
        s_time = time.time()
        model.eval()
        with torch.no_grad():
            valid_loss = []
            for img, mask, _, _ in eval_dataLoader:
                img = img.to(device=device, dtype=torch.float32)
                real_mask = mask.to(device=device, dtype=torch.float32)

                loss, predict_mask = model.eval_one_epoch(imgs=img, masks=real_mask)
                valid_loss.append(float(loss.item()))

                # show the image to Visdom
                if len(valid_loss) % 5 == 0:
                    model.show_mask(vis_train, img[0], mask[0], title="Ground Truth")
                    model.show_mask(vis_train, img[0], predict_mask[0], "Predicted")

        eval_loss = sum(valid_loss) / len(eval_dataLoader)
        loss_path_eval.append(eval_loss)
        fps = len(eval_dataLoader) / (time.time() - s_time)
        print('epoch {0} train_loss: {1:.6f} eval_loss: {2:.6f}'.format(epoch_i, train_loss, eval_loss))

    if loss_plot:
        print('**********FINISH**********')
        plt.title('Loss Performance for Few-Shot Learning')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim((0, 1))
        plt.plot(range(len(loss_path_train)), loss_path_train, color='blue', label='train')
        plt.plot(range(len(loss_path_eval)), loss_path_eval, color='yellow', label='eval')
        plt.legend()
        plt.savefig(os.path.join('{0}/figures'.format(root_path), 'few_shot ' + ".png"))
        plt.show()

    return train_best_loss


if __name__ == '__main__':
    device = "cuda:0"
    vis_train = visdom.Visdom(env="FewShot_Train")
    vis_eval = visdom.Visdom(env="FewShot_Evaluation")
    vis_pred = visdom.Visdom(env="FewShot_Prediction")

    # set hyperparameter list
    best_hyperparameters = {
        "lr": 5e-5,
        "weight_decay": 5e-5,
        "scheduler": 0.97
    }
    hyperparameters_grids = {'lr': [5e-5], 'weight_decay': [5e-5], 'scheduler': [0.97], }
    hyperparameters_sets = product(hyperparameters_grids['lr'], hyperparameters_grids['weight_decay'],
                                   hyperparameters_grids['scheduler'], shuffle=True)

    categories = classified_dataloader.get_categories(flag='labeled')
    category_loaders = []
    batch_sum = 0
    for c in categories:
        category_loaders.append(classified_dataloader.SitesLoader(config.DataLoaderConfig, c, flag="labeled"))
        batch_sum += len(categories[-1])

    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    print('Labeled data batch amount: {0}, evaluation data batch amount: {1}'.format(batch_sum, len(eval_dataLoader)))

    best_loss = 100
    for (_lr, _weight_decay, _scheduler) in hyperparameters_sets[:18]:
        loss = train(None, _lr, _weight_decay, _scheduler, category_loaders, eval_dataLoader, epoch_num=50,
                     loss_plot=True)
        print(
            "    Model loss (hyperparameter tunning) for lr={0}: {1:.4f}".format(_lr, loss))
        if loss < best_loss:
            best_loss = loss
            best_hyperparameters = {
                "lr": _lr,
                "weight_decay": _weight_decay,
                "scheduler": _scheduler,
            }
