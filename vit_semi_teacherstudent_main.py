# @Time : 2023/6/20 15:09
# @Author : Li Jiaqi
# @Description :
import os.path

import archaeological_georgia_biostyle_dataloader
import torch
import config
import visdom
import models.Loss as myLoss
import matplotlib.pyplot as plt
import time

from models.VitSegModel import VitSegModel

PESUDO_MAKS_THRESHOLD = 0.7
CONFIDENT_THRESHOLD = 0.7

supervise_loss_weight = 0.7
self_supervise_loss_weight = 0.3
# python -m visdom.server

def threshold_pseudo_masks(img, masks):
    N = masks.size(0)
    masks_flat = masks.squeeze(dim=1)
    masks_flat = masks_flat.view(N, -1)
    pixel_num = torch.sum(torch.abs(masks_flat), dim=1)
    confidence = torch.where((masks_flat >= PESUDO_MAKS_THRESHOLD) | (masks_flat <= 1 - PESUDO_MAKS_THRESHOLD), 1, 0)
    # confidence = torch.where((masks_flat >= PESUDO_MAKS_THRESHOLD) | (masks_flat <= 1 - PESUDO_MAKS_THRESHOLD), 1, 0)
    confidence = torch.sum(confidence, dim=1) / torch.numel(masks[0])

    pseudo_mask = torch.where((masks >= PESUDO_MAKS_THRESHOLD), 1, 0)
    # pseudo_mask = torch.as_tensor(masks >= PESUDO_MAKS_THRESHOLD, dtype=torch.int32)

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


if __name__ == '__main__':
    vis_teacher = visdom.Visdom(env='teacher')
    vis_student = visdom.Visdom(env='student')
    vis_eval = visdom.Visdom(env='eval')

    unlabel_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig,
                                                                                flag="pseudo")
    label_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="train")
    eval_dataLoader = archaeological_georgia_biostyle_dataloader.SitesLoader(config.DataLoaderConfig, flag="eval")
    print('Labeled data batch amount: ', len(unlabel_dataLoader) + len(label_dataLoader))

    teacher_model = VitSegModel()
    student_model = VitSegModel()
    loss_function = myLoss.SegmentationLoss(1, loss_type='dice', activation='none')

    loss_path_train = []
    loss_path_eval = []
    best_loss = 100
    for epoch_i in range(config.ModelConfig['epoch_num']):
        epoch_loss = []
        teacher_model.train()
        student_model.train()
        for img, _, _, _ in unlabel_dataLoader:
            img = img.to(device="cuda:0", dtype=torch.float32)
            predicted_masks = teacher_model.predict(img)
            confident_img, confident_mask, confident_predicted, confidence = \
                threshold_pseudo_masks(img, predicted_masks)
            teacher_loss_pseudo = 0
            if confident_img is not None:
                teacher_loss_pseudo = loss_function(confident_predicted, confident_mask)
                teacher_model.train_from_loss(teacher_loss_pseudo)
                teacher_model.show_mask(vis_teacher, confident_img[0], confident_predicted[0])
                teacher_model.show_mask(vis_teacher, confident_img[0], confident_mask[0])
            print('teacher_pseudo_loss:', float(teacher_loss_pseudo))

        for img, ground_truth, _, _ in label_dataLoader:
            img = img.to(device="cuda:0", dtype=torch.float32)
            ground_truth = ground_truth.to(device="cuda:0", dtype=torch.float32)
            ground_truth = ground_truth.unsqueeze(1)
            # train teacher one epoch
            teacher_loss_gt = teacher_model.train_one_epoch(img, ground_truth)
            # predict from the teacher
            teacher_predicted_masks = teacher_model.predict(img)
            # predict from the student
            student_predicted_masks = student_model.predict(img)
            # learn from both teacher and ground truth
            student_loss = loss_function(student_predicted_masks, ground_truth)
            self_supervise_loss = loss_function(student_predicted_masks, teacher_predicted_masks)
            loss = supervise_loss_weight * student_loss + self_supervise_loss_weight * self_supervise_loss
            student_model.train_from_loss(loss)
            epoch_loss.append(loss)

            # show results
            student_model.show_mask(vis_student, img[0], ground_truth[0])
            student_model.show_mask(vis_student, img[0], teacher_predicted_masks[0])
            student_model.show_mask(vis_student, img[0], student_predicted_masks[0])
            print('summation loss:{0:.3f} teacher loss: {1:.3f} student loss: {2:.3f} self-supervised loss:{3:.3f}'
                  .format(float(loss), float(teacher_loss_gt), float(student_loss), float(self_supervise_loss)))
        train_loss = sum(epoch_loss) / len(label_dataLoader)
        teacher_model.scheduler_step()
        student_model.scheduler_step()

        # eval
        s_time = time.time()
        student_model.eval()
        with torch.no_grad():
            valid_loss = []
            for img, mask, _, _ in eval_dataLoader:
                img = img.to(device="cuda:0", dtype=torch.float32)
                real_mask = mask.to(device="cuda:0", dtype=torch.float32)
                real_mask = real_mask.unsqueeze(1)
                predict_mask = student_model.predict(img)
                loss = loss_function(predict_mask, real_mask)
                valid_loss.append(float(loss.item()))

                # show the image to Visdom
                if len(valid_loss) % 2 == 0:
                    student_model.show_mask(vis_eval, img[0], real_mask[0])
                    student_model.show_mask(vis_eval, img[0], predict_mask[0])

        eval_loss = sum(valid_loss) / len(eval_dataLoader)
        fps = len(eval_dataLoader) / (time.time() - s_time)

        # checkpoint
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(student_model.state_dict(),
                       os.path.join('checkpoints',
                                    'self-pseudo vit-seg epoch {0} train {1:.3f} eval {2:.3f} fps {3:.2f}.pth'
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
