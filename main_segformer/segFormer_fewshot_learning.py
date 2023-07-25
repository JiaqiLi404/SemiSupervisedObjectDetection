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

def prediction(weight):
    model = SegModel(weight, lr=best_hyperparameters['lr'],
                     weight_decay=best_hyperparameters['weight_decay'],
                     scheduler=best_hyperparameters['scheduler'])
    model.add_cls_token()
    for img, mask, _, _ in eval_dataLoader:
        img = img.to(device=device, dtype=torch.float32)
        real_mask = mask.to(device=device, dtype=torch.float32)

        loss, predict_mask = model.eval_one_epoch(imgs=img, masks=real_mask)

        # show the image to Visdom
        for img_i in range(img.shape[0]):
            model.show_mask(vis_eval, img[img_i], mask[img_i], title="Ground Truth")
            model.show_mask(vis_eval, img[img_i], predict_mask[img_i], "Predicted")


def train(pretrain_weight, _lr, _weight_decay, _scheduler, category_dataloaders, eval_dataLoader,
          epoch_num=config.ModelConfig['epoch_num'], iteration_num=35, save_model=False, loss_plot=None):
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

        for loader in category_dataloaders:
            loader.reshuffle()

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
            category_1_loss, category_1_predicted, category_1_cls_token = model.predict(category_1_img, category_1_mask,
                                                                                        output_cls_token=True,
                                                                                        isEval=False)
            category_2_loss, category_2_predicted, category_2_cls_token = model.predict(category_2_img, category_2_mask,
                                                                                        output_cls_token=True,
                                                                                        isEval=False)

            # batch_size = min(category_1_cls_token.shape[0], category_2_cls_token.shape[0])
            # # inter-loss
            # inter_loss = compute_similarity(category_1_cls_token[:batch_size, :, :],
            #                                 category_2_cls_token[:batch_size, :, :])
            # # intra-loss
            # intra_loss_1 = 1 - compute_similarity(category_1_cls_token[:batch_size // 2, :, :],
            #                                       category_1_cls_token[-(batch_size // 2):, :, :])
            # intra_loss_2 = 1 - compute_similarity(category_2_cls_token[:batch_size // 2, :, :],
            #                                       category_2_cls_token[-(batch_size // 2):, :, :])
            # # intra_loss = (intra_loss_1 + intra_loss_2) / 2
            #
            # category_1_summation_loss = (category_1_loss + inter_loss + intra_loss_1) / 3
            # category_2_summation_loss = (category_2_loss + inter_loss + intra_loss_2) / 3

            intra_loss_1 = 0
            intra_loss_2 = 0
            inter_loss = 0
            category_1_summation_loss = category_1_loss
            category_2_summation_loss = category_2_loss
            summation_loss = (category_1_summation_loss + category_2_summation_loss) / 2

            loss_categories[category_1] = loss_categories[category_1] * 0.5 + category_1_summation_loss * 0.5
            loss_categories[category_2] = loss_categories[category_2] * 0.5 + category_2_summation_loss * 0.5

            model.train_from_loss(summation_loss)
            epoch_loss.append(float(summation_loss))

            # show results
            if len(epoch_loss) % 5 == 0:
                model.show_mask(vis_train, category_1_img[0], category_1_mask[0], title="Ground Truth")
                model.show_mask(vis_train, category_1_img[0], category_1_predicted[0], "Predicted")
                print(
                    "summation loss:{0:.3f} cat_1_sum_loss:{1:.3f} cat_1_cls_loss:{2:.3f} cat_1_intra_loss:{3:.3f} inter_loss:{4:.3f}".format(
                        epoch_loss[-1], category_1_summation_loss, category_1_loss, intra_loss_1, inter_loss))
                print(
                    "summation loss:{0:.3f} cat_2_sum_loss:{1:.3f} cat_2_cls_loss:{2:.3f} cat_2_intra_loss:{3:.3f} inter_loss:{4:.3f}".format(
                        epoch_loss[-1], category_2_summation_loss, category_2_loss, intra_loss_2, inter_loss))
                print(' ')

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
        print('epoch {0} train_loss: {1:.6f} eval_loss: {2:.6f}\n'.format(epoch_i, train_loss, eval_loss))

        # checkpoint
        if eval_loss < train_best_loss:
            train_best_loss = eval_loss
            if save_model:
                torch.save(model.state_dict(),
                           os.path.join('../checkpoints',
                                        'few-shot seg-former epoch {0} train {1:.3f} eval {2:.3f} fps {3:.2f}.pth'
                                        .format(epoch_i, train_loss, eval_loss, fps)))

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


def compute_similarity(mat1, mat2):
    mat1 = mat1.squeeze(1)
    mat2 = mat2.squeeze(1)
    return torch.mean(similarity_loss(mat1, mat2))


def train_autoencoder_iteration(model, category_dataloaders, category_dataloaders_iter):
    # randomly pick two categories
    [category_1, category_2] = random.sample(list(range(len(category_dataloaders))), 2)
    try:
        category_1_img, _ = next(category_dataloaders_iter[category_1])
    except StopIteration:
        category_dataloaders_iter[category_1] = iter(category_dataloaders[category_1])
        category_1_img, _ = next(category_dataloaders_iter[category_1])

    try:
        category_2_img, _ = next(category_dataloaders_iter[category_2])
    except StopIteration:
        category_dataloaders_iter[category_2] = iter(category_dataloaders[category_2])
        category_2_img, _ = next(category_dataloaders_iter[category_2])

    category_1_img = category_1_img.to(device=device, dtype=torch.float32)
    category_2_img = category_2_img.to(device=device, dtype=torch.float32)

    # supervised loss
    category_1_loss, category_1_predicted, category_1_cls_token = model.predict(category_1_img, category_1_img,
                                                                                output_cls_token=True,
                                                                                isEval=False, use_loss='mse')
    category_2_loss, category_2_predicted, category_2_cls_token = model.predict(category_2_img, category_2_img,
                                                                                output_cls_token=True,
                                                                                isEval=False, use_loss='mse')

    batch_size = min(category_1_cls_token.shape[0], category_2_cls_token.shape[0])
    # inter-loss
    inter_loss = 0.5 + 0.5*compute_similarity(category_1_cls_token[:batch_size, :, :],
                                    category_2_cls_token[:batch_size, :, :])
    # intra-loss
    intra_loss_1 = 0.5 - 0.5*compute_similarity(category_1_cls_token[:batch_size // 2, :, :],
                                          category_1_cls_token[-(batch_size // 2):, :, :])
    intra_loss_2 = 0.5 - 0.5*compute_similarity(category_2_cls_token[:batch_size // 2, :, :],
                                          category_2_cls_token[-(batch_size // 2):, :, :])
    # intra_loss = (intra_loss_1 + intra_loss_2) / 2

    category_1_summation_loss = category_1_loss + 100 * inter_loss + 100 * intra_loss_1
    category_2_summation_loss = category_2_loss + 100 * inter_loss + 100 * intra_loss_2

    # intra_loss_1 = 0
    # intra_loss_2 = 0
    # inter_loss = 0
    # category_1_summation_loss = category_1_loss
    # category_2_summation_loss = category_2_loss
    summation_loss = (category_1_summation_loss + category_2_summation_loss) / 2
    return summation_loss, category_1_img, category_1_predicted, category_1_loss, category_2_loss, intra_loss_1, intra_loss_2, inter_loss


def train_autoencoder(pretrain_weight, _lr, _weight_decay, _scheduler, category_dataloaders1, category_dataloaders2,
                      eval_dataLoader, epoch_num=config.ModelConfig['epoch_num'], iteration_num=101, save_model=False,
                      loss_plot=None):
    print('**************** Train *******************')
    print('lr: {0}'.format(_lr))
    model = SegModel(pretrain_weight, _lr, _weight_decay, _scheduler, num_labels=3)
    model.add_cls_token()

    loss_path_train = []
    loss_path_eval = []
    train_best_loss = 100
    train_best_epoch = 0
    category_dataloaders_1_iter = []
    category_dataloaders_2_iter = []
    for loader in category_dataloaders1:
        category_dataloaders_1_iter.append(iter(loader))
    for loader in category_dataloaders2:
        category_dataloaders_2_iter.append(iter(loader))

    for epoch_i in range(epoch_num):
        for loader in category_dataloaders1:
            loader.reshuffle()
        for loader in category_dataloaders2:
            loader.reshuffle()

        epoch_loss = []
        model.train()

        for iter_i in range(iteration_num):
            category_1_summation_loss, category_1_img, category_1_predicted, category_1_loss, category_2_loss, intra_loss_1, intra_loss_2, inter_loss = train_autoencoder_iteration(
                model,
                category_dataloaders1,
                category_dataloaders_1_iter)
            category_2_summation_loss, category_2_img, category_2_predicted, category_1_loss, category_2_loss, intra_loss_1, intra_loss_2, inter_loss = train_autoencoder_iteration(
                model,
                category_dataloaders2,
                category_dataloaders_2_iter)
            summation_loss = (category_1_summation_loss + category_2_summation_loss) / 2

            model.train_from_loss(summation_loss)
            epoch_loss.append(float(summation_loss))

            # show results
            if len(epoch_loss) % 20 == 0:
                model.show_mask(vis_train, category_1_img[0], None, "Ground Truth")
                model.show_mask(vis_train, category_1_predicted[0].detach(), None, "Predicted")
                print(
                    "summation loss:{0:.3f} cat_1_sum_loss:{1:.3f} cat_1_cls_loss:{2:.3f} cat_1_intra_loss:{3:.3f} inter_loss:{4:.3f}".format(
                        epoch_loss[-1], category_1_summation_loss, category_1_loss, intra_loss_1, inter_loss))
                print(
                    "summation loss:{0:.3f} cat_2_sum_loss:{1:.3f} cat_2_cls_loss:{2:.3f} cat_2_intra_loss:{3:.3f} inter_loss:{4:.3f}".format(
                        epoch_loss[-1], category_2_summation_loss, category_2_loss, intra_loss_2, inter_loss))
                print(' ')

        train_loss = sum(epoch_loss) / iteration_num
        loss_path_train.append(train_loss)
        model.scheduler_step()

        # eval
        s_time = time.time()
        model.eval()
        with torch.no_grad():
            valid_loss = []
            for img, _, _, _ in eval_dataLoader:
                img = img.to(device=device, dtype=torch.float32)

                category_1_loss, category_1_predicted, category_1_cls_token = model.predict(img,
                                                                                            img,
                                                                                            output_cls_token=True,
                                                                                            isEval=True,
                                                                                            use_loss='mse')
                valid_loss.append(float(category_1_loss.item()))

                # show the image to Visdom
                if len(valid_loss) % 5 == 0:
                    model.show_mask(vis_train, img[0], None, "Ground Truth")
                    model.show_mask(vis_train, category_1_predicted[0], None, "Predicted")

        eval_loss = sum(valid_loss) / len(eval_dataLoader)
        loss_path_eval.append(eval_loss)
        fps = len(eval_dataLoader) / (time.time() - s_time)
        print('epoch {0} train_loss: {1:.6f} eval_loss: {2:.6f}\n'.format(epoch_i, train_loss, eval_loss))

        # checkpoint
        if eval_loss < train_best_loss:
            train_best_loss = eval_loss
            if save_model:
                torch.save(model.state_dict(),
                           os.path.join('../checkpoints',
                                        'few-shot seg-former epoch {0} train {1:.3f} eval {2:.3f} fps {3:.2f}.pth'
                                        .format(epoch_i, train_loss, eval_loss, fps)))

    if loss_plot:
        print('**********FINISH**********')
        plt.title('Loss Performance for Few-Shot Learning')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim((0, 500))
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

    similarity_loss = torch.nn.CosineSimilarity(dim=1)

    # set hyperparameter list : 5e-5
    best_hyperparameters = {
        "lr": 2e-5,
        "weight_decay": 5e-5,
        "scheduler": 0.97
    }
    hyperparameters_grids = {'lr': [8e-5, 5e-5, 2e-5, 5e-6], 'weight_decay': [5e-5], 'scheduler': [0.97], }
    hyperparameters_sets = product(hyperparameters_grids['lr'], hyperparameters_grids['weight_decay'],
                                   hyperparameters_grids['scheduler'], shuffle=True)

    categories = classified_dataloader.get_categories(flag='labeled')
    category_loaders_labeled = []
    batch_sum_labeled = 0
    for c in categories:
        category_loaders_labeled.append(classified_dataloader.SitesLoader(config.DataLoaderConfig, c, flag="labeled"))
        batch_sum_labeled += len(categories[-1])

    categories = classified_dataloader.get_categories(flag='unlabeled')
    category_loaders_unlabeled = []
    batch_sum_unlabeled = 0
    for c in categories:
        category_loaders_unlabeled.append(
            classified_dataloader.SitesLoader(config.DataLoaderConfig, c, flag="unlabeled"))
        batch_sum_unlabeled += len(categories[-1])

    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    print('Labeled data batch amount: {0}, Unlabeled data batch amount: {1}, evaluation data batch amount: {2}'.format(
        batch_sum_labeled, batch_sum_unlabeled, len(eval_dataLoader)))

    best_loss = 100
    # for (_lr, _weight_decay, _scheduler) in hyperparameters_sets[:18]:
    #     loss = train_autoencoder(None, _lr, _weight_decay, _scheduler, category_loaders_labeled,
    #                              category_loaders_unlabeled, eval_dataLoader, epoch_num=20,
    #                              loss_plot=True, save_model=False)
    #     print(
    #         "    Model loss (hyperparameter tunning) for lr={0}: {1:.4f}".format(_lr, loss))
    #     if loss < best_loss:
    #         best_loss = loss
    #         best_hyperparameters = {
    #             "lr": _lr,
    #             "weight_decay": _weight_decay,
    #             "scheduler": _scheduler,
    #         }

    # train the domain prompt autoencoder
    loss = train_autoencoder(None,
                             best_hyperparameters['lr'], best_hyperparameters['weight_decay'],
                             best_hyperparameters['scheduler'], category_loaders_labeled, category_loaders_unlabeled,
                             eval_dataLoader, epoch_num=200,
                             loss_plot=True, save_model=True)
    #
    # # train the improved SegFormer
    # train(None, best_hyperparameters['lr'], best_hyperparameters['weight_decay'], best_hyperparameters['scheduler'],
    #       category_loaders_labeled, eval_dataLoader, loss_plot=True, save_model=True)

    # prediction('few-shot without cls loss seg-former epoch 24 train 0.078 eval 0.308 fps 2.71.pth')
    # prediction('few-shot autoencoder prompt tuning segFormer_epoch_38_train_0.160_eval_0.330_fps_4.34.pth')
