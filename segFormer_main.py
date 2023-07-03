import os.path

import archaeological_georgia_biostyle_dataloader
import torch
import config
import visdom
import models.Loss as myLoss
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import copy
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from models.SegFormerModel import SegFormerModel
import numpy as np
from datasets import load_metric
from sklearn.model_selection import GridSearchCV
import math

visdom_display_freq = 5

def Prediction():
    vis_pred = visdom.Visdom(env="SegFormer_Prediction")
    pretained_model = "nvidia/mit-b5"
    visdom_display_freq = 5 # send image to visdom every 5 epoch
    # feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
    feature_extractor = SegformerFeatureExtractor.from_pretrained(pretained_model)
    unlabel_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                                flag="unlabeled")
    model = SegFormerModel() # with pre-trained weight
    model.eval()
    with torch.no_gard():
        dataPatches = 0
        for img, _, _, _ in unlabel_dataLoader:
            encoded_inputs = feature_extractor(img, return_tensors="pt")
            encoded_img = encoded_inputs["pixel_values"]
            encoded_img = encoded_img.to(device=device)

            predict_mask = model.predict(img=encoded_img)# logits are of shape (batch_size, num_labels, height/4, width/4)
            dataPatches += 1
            if dataPatches % visdom_display_freq == 0:
                model.show_mask(vis_pred, img[0], None, title="Raw Image {0}".format(dataPatches))
                model.show_mask(vis_pred, img[0], predict_mask[0], title="Predicted Mask epoch{0}".format(dataPatches))

def Train(model, train_dataloader, eval_dataLoader, feature_extractor, epoch_num=config.ModelConfig['epoch_num'], save_model = False, loss_plot = None, visdom_display = False):
    if visdom_display:
        vis_train = visdom.Visdom(env="SegFormer_Train")
        vis_eval = visdom.Visdom(env="SegFormer_Evaluation")
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
            encoded_inputs = feature_extractor(img, mask, return_tensors="pt")
            encoded_img = encoded_inputs["pixel_values"]
            encoded_mask = encoded_inputs["labels"]

            encoded_img = encoded_img.to(device=device)
            encoded_mask = encoded_mask.to(device=device)

            # forward
            loss, predict_mask = model.train_one_epoch(imgs=encoded_img, masks=encoded_mask)
            train_epoch_loss.append(loss.item())

            # does this metrics make sense?
            # results = metric.compute(predictions=predict_mask, references=mask, num_labels=2, reduce_labels=False, ignore_index=255)
            
            if visdom_display and len(train_epoch_loss) % visdom_display_freq == 0:
                model.show_mask(vis_train, img[0], mask[0], title="Ground Truth")
                model.show_mask(vis_train, img[0], predict_mask[0], title="Predicted Mask epoch{0}".format(epoch_i))
        
        train_loss = sum(train_epoch_loss) / len(train_dataloader)
        train_loss_path.append(train_loss)
        model.scheduler_step()

        # evaluation
        s_time = time.time()
        model.eval()
        eval_epoch_loss = []
        with torch.no_grad():
            for img, mask, _, _ in eval_dataLoader:
                encoded_inputs = feature_extractor(img, mask, return_tensors="pt")
                encoded_img = encoded_inputs["pixel_values"]
                encoded_mask = encoded_inputs["labels"]

                encoded_img = encoded_img.to(device=device)
                encoded_mask = encoded_mask.to(device=device)

                # forward
                loss, predict_mask = model.eval_one_epoch(imgs=encoded_img, masks=encoded_mask)
                eval_epoch_loss.append(loss.item())

                if visdom_display and len(eval_epoch_loss) % visdom_display_freq == 0:
                    model.show_mask(vis_eval, img[0], mask[0], title="Ground Truth")
                    model.show_mask(vis_eval, img[0], predict_mask[0], title="Predicted Mask epoch{0}".format(epoch_i))

        eval_loss = sum(eval_epoch_loss) / len(eval_dataLoader)
        eval_loss_path.append(eval_loss)
        fps = (time.time() - s_time) / len(eval_dataLoader)

        print('epoch {0} train_loss: {1:.6f} eval_loss: {2:.6f} fps {3:.2f}'.format(epoch_i, train_loss, eval_loss, fps))

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch_i
            if save_model:
                torch.save(model.state_dict(),
                            os.path.join('checkpoints',
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
        plt.savefig(os.path.join('figures', "_".join(loss_plot.split(" "))+".png"))
        plt.show()

    return best_loss, best_epoch

def Hyperparameter_Tuning(lr=[1e-5], weight_decay=[5e-5], scheduler=[0.97]):
    pretained_model = "nvidia/mit-b5"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(pretained_model)

    label_dataset = archaeological_georgia_biostyle_dataloader.SitesBingBook(config.DataLoaderConfig["dataset"], config.DataLoaderConfig["maskdir"], config.DataLoaderConfig["transforms"])
    train_data_num = math.floor(len(label_dataset) * 0.8)
    train_dataset, validation_dataset = torch.utils.data.random_split(label_dataset, [train_data_num, len(label_dataset)-train_data_num])
    train_dataloader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, dataset=train_dataset, flag="train")
    validation_dataloader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, dataset=validation_dataset, flag="train")

    print('Training data batch amount: {0}, Validation data batch amount: {1}'.format(len(train_dataloader), len(validation_dataloader)))

    best_loss = 100
    best_hyperparameters = {
        "lr": None,
        "weight_decay": None,
        "scheduler": None
    }
    for _lr in lr:
        for _weight_decay in weight_decay:
            for _scheduler in scheduler:
                print("Training model (hyperparameter tunning) for lr={0}, weight_decay={1}, scheduler={2}".format(_lr, _weight_decay, _scheduler))
                model = SegFormerModel(lr=_lr, weight_decay=_weight_decay, scheduler=_scheduler)
                loss, trained_epoch = Train(model, train_dataloader, validation_dataloader, save_model=False)
                print("    Model loss (hyperparameter tunning) for lr={0}, weight_decay={1}, scheduler={2}: {3:.4f}".format(_lr, _weight_decay, _scheduler, loss))
                if loss < best_loss:
                    best_loss = loss
                    best_hyperparameters = {
                        "lr": _lr,
                        "weight_decay": _weight_decay,
                        "scheduler": _scheduler
                    }
    
    return best_hyperparameters

if __name__ == '__main__':
    device="cuda:0"
    # feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)

    best_hyperparameters = {
        "lr": 1e-5,
        "weight_decay": 5e-5,
        "scheduler": 0.97
    }
    # set hyperparameter list
    best_hyperparameters = Hyperparameter_Tuning(lr=[1e-5], weight_decay=[5e-5], scheduler=[0.97])

    pretained_model = "nvidia/mit-b5"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(pretained_model)
    label_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="train")
    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    print('Labeled data batch amount: {0}, evaluation data batch amount: {1}'.format(len(label_dataLoader), len(eval_dataLoader)))

    print("Training model for lr={0}, weight_decay={1}, scheduler={2}".format(best_hyperparameters['lr'], best_hyperparameters['weight_decay'], best_hyperparameters['scheduler']))
    model = SegFormerModel(lr=best_hyperparameters['lr'], weight_decay=best_hyperparameters['weight_decay'], scheduler=best_hyperparameters['scheduler'])
    Train(model, label_dataLoader, eval_dataLoader, feature_extractor, save_model = True, visdom_display=True, loss_plot="Loss Performance of SegFormer")
