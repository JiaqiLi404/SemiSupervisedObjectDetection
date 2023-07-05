# @Time : 2023/6/20 15:09
# @Author : Li Jiaqi
# @Description :
import os.path

import archaeological_georgia_biostyle_dataloader
import torch
import config
import visdom
import matplotlib.pyplot as plt
import time
from Utils import product
import math
import models.Loss as myLoss

from models.SegFormerModel import SegFormerModel as SegModel

PESUDO_MAKS_THRESHOLD = 0.7
CONFIDENT_THRESHOLD = 0.7


# python -m visdom.server

def threshold_pseudo_masks(img, masks):
    N = masks.size(0)
    # masks = masks.squeeze(dim=1)
    masks_flat = masks.reshape(N, -1)
    pixel_num = torch.sum(torch.abs(masks_flat), dim=1)
    confidence = torch.where((masks_flat >= PESUDO_MAKS_THRESHOLD) | (masks_flat <= 1 - PESUDO_MAKS_THRESHOLD), 1, 0)
    # confidence = torch.where((masks_flat >= PESUDO_MAKS_THRESHOLD) | (masks_flat <= 1 - PESUDO_MAKS_THRESHOLD), 1, 0)
    confidence = torch.sum(confidence, dim=1) / torch.numel(masks[0])

    pseudo_mask = torch.where((masks >= PESUDO_MAKS_THRESHOLD), 1, 0)
    # pseudo_mask = torch.as_tensor(masks >= PESUDO_MAKS_THRESHOLD, dtype=torch.int32)

    target_num = torch.sum(torch.where(masks_flat >= PESUDO_MAKS_THRESHOLD, 1, 0), dim=1)

    confident_img = []
    confident_mask = []
    confident_predicted = []
    for n in range(N):
        if pixel_num[n] > 1000 and confidence[n] >= CONFIDENT_THRESHOLD:
            confident_img.append(img[n])
            confident_predicted.append(masks[n])
            confident_mask.append(pseudo_mask[n])
    if len(confident_img) != 0:
        confident_img = torch.stack(confident_img)
        confident_predicted = torch.stack(confident_predicted)
        confident_mask = torch.stack(confident_mask)
    else:
        confident_img = confident_mask = confident_predicted = None

    return confident_img, confident_mask, confident_predicted, confidence


def train(pretrain_weight, teacher_lr, student_lr, weight_decay, scheduler, supervise_weight, this_eval_dataloader,
          epochs=config.ModelConfig['epoch_num'],
          save_checkpoints=False, plot_loss=False):
    print('**************** Train *******************')
    print('teacher_lr: {0} student_lr: {1} supervise_weight: {2} threshold: {3}'.format(teacher_lr, student_lr,
                                                                                        supervise_weight,
                                                                                        PESUDO_MAKS_THRESHOLD))
    teacher_model = SegModel(pretrain_weight, teacher_lr, weight_decay, scheduler)
    student_model = SegModel(pretrain_weight, student_lr, weight_decay, scheduler)

    loss_path_train = []
    loss_path_eval = []
    loss_path_train_teacher = []
    loss_path_eval_teacher = []
    best_loss = 100
    for epoch_i in range(epochs):
        epoch_loss = []
        epoch_loss_teacher = []
        teacher_model.train()
        student_model.train()
        image_used = 0
        for img, _, _, _ in unlabel_dataLoader:
            with torch.no_grad():
                img = img.to(device=device, dtype=torch.float32)
                predicted_masks = teacher_model.predict(img)
                confident_img, confident_mask, confident_predicted, confidence = \
                    threshold_pseudo_masks(img, predicted_masks)
                teacher_loss_pseudo = 0
            if confident_img is not None:
                image_used += confident_img.size(0)
                confident_mask = confident_mask.squeeze(1)
                teacher_loss_pseudo, confident_predicted = teacher_model.train_one_epoch(confident_img, confident_mask)
                teacher_model.show_mask(vis_teacher, confident_img[0], confident_predicted[0],
                                        title="Teacher Predict epoch{0}".format(epoch_i))
                teacher_model.show_mask(vis_teacher, confident_img[0], confident_mask[0],
                                        title="Teacher Pseudo Mask epoch{0}".format(epoch_i))
            print('teacher_pseudo_loss:', float(teacher_loss_pseudo))

        print('epoch {0}: {1} unlabeled images used'.format(epoch_i, image_used))

        for img, ground_truth, _, _ in label_dataLoader:
            img = img.to(device=device, dtype=torch.float32)
            ground_truth = ground_truth.to(device=device, dtype=torch.float32)
            # ground_truth = ground_truth.unsqueeze(1)
            # train teacher one epoch
            teacher_loss_gt, _ = teacher_model.train_one_epoch(img, ground_truth)
            # predict from the teacher
            with torch.no_grad():
                teacher_predicted_masks = teacher_model.predict(img)
            # predict from the student
            student_loss, student_predicted_masks = student_model.predict(img, ground_truth)
            # learn from both teacher and ground truth

            self_supervise_loss = loss_func(student_predicted_masks, teacher_predicted_masks)
            loss = supervise_weight * student_loss + (1 - supervise_weight) * self_supervise_loss
            student_model.train_from_loss(loss)
            epoch_loss.append(float(loss.item()))
            epoch_loss_teacher.append(float(teacher_loss_gt.item()))

            # show results
            if len(epoch_loss) % 4 == 0:
                student_model.show_mask(vis_student, img[0], ground_truth[0], title="Ground Truth")
                student_model.show_mask(vis_student, img[0], teacher_predicted_masks[0],
                                        title="Teacher Predict (train) epoch{0}".format(epoch_i))
                student_model.show_mask(vis_student, img[0], student_predicted_masks[0],
                                        title="Student Predict (train) epoch{0}".format(epoch_i))
                print('summation loss:{0:.3f} teacher loss: {1:.3f} student loss: {2:.3f} self-supervised loss:{3:.3f}'
                      .format(float(loss), float(teacher_loss_gt), float(student_loss), float(self_supervise_loss)))

        train_loss = sum(epoch_loss) / len(label_dataLoader)
        train_loss_teacher = sum(epoch_loss_teacher) / len(label_dataLoader)
        teacher_model.scheduler_step()
        student_model.scheduler_step()

        # eval
        s_time = time.time()
        student_model.eval()
        teacher_model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_loss_teacher = []
            for img, mask, _, _ in this_eval_dataloader:
                img = img.to(device=device, dtype=torch.float32)
                real_mask = mask.to(device=device, dtype=torch.float32)

                teacher_loss, predict_mask_teacher = teacher_model.eval_one_epoch(imgs=img, masks=real_mask)
                valid_loss_teacher.append(float(teacher_loss.item()))
                student_loss, predict_mask = student_model.eval_one_epoch(imgs=img, masks=real_mask)
                valid_loss.append(float(student_loss.item()))

                # show the image to Visdom
                if len(valid_loss) % 2 == 0:
                    student_model.show_mask(vis_eval, img[0], real_mask[0], title="Ground Truth")
                    student_model.show_mask(vis_eval, img[0], predict_mask_teacher[0],
                                            title="Teacher Predict (eval) epoch{0}".format(epoch_i))
                    student_model.show_mask(vis_eval, img[0], predict_mask[0],
                                            title="Student Predict (eval) epoch{0}".format(epoch_i))

        eval_loss = sum(valid_loss) / len(eval_dataLoader)
        eval_loss_teacher = sum(valid_loss_teacher) / len(eval_dataLoader)
        fps = len(eval_dataLoader) / (time.time() - s_time)

        # checkpoint
        if eval_loss < best_loss:
            best_loss = eval_loss
            if save_checkpoints:
                torch.save(student_model.state_dict(), os.path.join('checkpoints',
                                                                    'self-pseudo seg-former student epoch {0} train {1:.3f} eval {2:.3f} fps {3:.2f}.pth'
                                                                    .format(epoch_i, train_loss, best_loss, fps)))
        if eval_loss_teacher < best_loss:
            best_loss = eval_loss_teacher
            if save_checkpoints:
                torch.save(teacher_model.state_dict(), os.path.join('checkpoints',
                                                                    'self-pseudo seg-former teacher epoch {0} train {1:.3f} eval {2:.3f} fps {3:.2f}.pth'
                                                                    .format(epoch_i, train_loss, best_loss, fps)))

        print(
            'epoch {0} train_loss: {1:.6f} eval_loss: {2:.6f} teacher_loss:{3:.6f} teacher_eval_loss:{4:.6f} fps {5:.2f}'
            .format(
                epoch_i,
                train_loss,
                eval_loss,
                train_loss_teacher,
                eval_loss_teacher,
                fps))
        loss_path_train.append(train_loss)  # .cpu().detach()
        loss_path_eval.append(eval_loss)
        loss_path_train_teacher.append(train_loss_teacher)
        loss_path_eval_teacher.append(eval_loss_teacher)

    if plot_loss:
        title = 't_lr-{0} s_lr-{1} supervise_w-{2} threshold-{3}' \
            .format(teacher_lr, student_lr, supervise_weight, PESUDO_MAKS_THRESHOLD)
        print('**********FINISH**********')
        plt.title('Loss Performance of SegFormer-Pseudo (Student)')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim((0, 1))
        plt.plot(range(epochs), loss_path_train, color='blue', label='train')
        plt.plot(range(epochs), loss_path_eval, color='yellow', label='eval')
        plt.legend()
        plt.savefig(os.path.join('figures', 'Loss Performance of SegFormer-Pseudo Student ' + title + '.png'))
        plt.show()

        plt.title('Loss Performance of SegFormer-Pseudo (Teacher)')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim((0, 1))
        plt.plot(range(epochs), loss_path_train_teacher, color='blue', label='train')
        plt.plot(range(epochs), loss_path_eval_teacher, color='yellow', label='eval')
        plt.legend()
        plt.savefig(os.path.join('figures', 'Loss Performance of SegFormer-Pseudo Teacher ' + title + '.png'))
        plt.show()

        plt.title('Loss Performance of SegFormer-Pseudo (Teacher-Student)')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim((0, 1))
        plt.plot(range(epochs), loss_path_train, color='red', label='student-train')
        plt.plot(range(epochs), loss_path_eval, color='orange', label='student-eval')
        plt.plot(range(epochs), loss_path_train_teacher, color='green', label='teacher-train')
        plt.plot(range(epochs), loss_path_eval_teacher, color='blue', label='teacher-eval')
        plt.legend()
        plt.savefig(os.path.join('figures', 'Loss Performance of SegFormer-Pseudo Teacher-Student ' + title + '.png'))
        plt.show()

    return best_loss


if __name__ == '__main__':
    pretrain_weight = 'segFormer_baseline_epoch_18_train_0.203_eval_0.322_fps_0.52.pth'
    device = "cuda:0"
    vis_teacher = visdom.Visdom(env='teacher')
    vis_student = visdom.Visdom(env='student')
    vis_eval = visdom.Visdom(env='eval')

    loss_func = myLoss.SegmentationLoss(1, loss_type='dice', activation='none')

    label_dataset = archaeological_georgia_biostyle_dataloader.SitesBingBook(config.DataLoaderConfig["dataset"],
                                                                             config.DataLoaderConfig["maskdir"],
                                                                             config.DataLoaderConfig["transforms"])
    train_data_num = math.floor(len(label_dataset) * 0.8)
    train_dataset, validation_dataset = torch.utils.data.random_split(label_dataset, [train_data_num,
                                                                                      len(label_dataset) - train_data_num])
    label_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                              dataset=train_dataset, flag="train")
    validation_dataloader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                                   dataset=validation_dataset,
                                                                                   flag="train")
    unlabel_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                                flag="pseudo")
    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    print('Labeled data batch amount: ', len(unlabel_dataLoader) + len(label_dataLoader))

    hyperparameters_grids = {'lr': [2e-5, 5e-6, 1e-6], 'weight_decay': [5e-5], 'scheduler': [0.97],
                             'supervise_loss_weight': [0.9], 'threshold': [0.7, 0.8, 0.85]}
    hyperparameters_sets = product(hyperparameters_grids['lr'], hyperparameters_grids['lr'],
                                   hyperparameters_grids['weight_decay'], hyperparameters_grids['scheduler'],
                                   hyperparameters_grids['supervise_loss_weight'], hyperparameters_grids['threshold'],
                                   shuffle=True)

    best_loss = 100
    best_hyperparameters = {
        "t_lr": 5e-6,
        "s_lr": 4e-5,
        "weight_decay": 5e-5,
        "scheduler": 0.97,
        'supervise_loss_weight': 0.9,
        'threshold': 0.8
    }
    for (_t_lr, _s_lr, _weight_decay, _scheduler, _supervise_weight, _threshold) in hyperparameters_sets[:4]:
        PESUDO_MAKS_THRESHOLD = _threshold
        loss = train(pretrain_weight, _t_lr, _s_lr, _weight_decay, _scheduler, _supervise_weight, validation_dataloader,
                     epochs=10, plot_loss=True)
        print(
            "    Model loss (hyperparameter tunning) for teacher_lr={0}, tstudent_lr={1}, supervise_weight={2}, threshold={3}: {4:.4f}".format(
                _t_lr, _s_lr, _supervise_weight, _threshold, loss))
        if loss < best_loss:
            best_loss = loss
            best_hyperparameters = {
                "t_lr": _t_lr,
                "s_lr": _s_lr,
                "weight_decay": _weight_decay,
                "scheduler": _scheduler,
                'supervise_loss_weight': _supervise_weight,
                'threshold': _threshold
            }

    loss = train(pretrain_weight, best_hyperparameters['t_lr'], best_hyperparameters['s_lr'],
                 best_hyperparameters['weight_decay'], best_hyperparameters['scheduler'],
                 best_hyperparameters['supervise_loss_weight'], eval_dataLoader, save_checkpoints=True, plot_loss=True,
                 epochs=1)
