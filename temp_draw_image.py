# @Time : 2023/7/5 18:36
# @Author : Li Jiaqi
# @Description :
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # T_lr 5e-6 s_lr 4e-5 ssw 0.1
    loss_path_train = [0.416, 0.370, 0.328, 0.291, 0.299, 0.256, 0.278, 0.245, 0.224, 0.210]
    loss_path_eval = [0.223, 0.325, 0.281, 0.326, 0.269, 0.240, 0.296, 0.207, 0.224, 0.209]
    loss_path_train_teacher = [0.196, 0.212, 0.204, 0.188, 0.189, 0.205, 0.226, 0.204, 0.220, 0.223]
    loss_path_eval_teacher = [0.173, 0.190, 0.182, 0.167, 0.174, 0.182, 0.194, 0.159, 0.167, 0.177]

    # # ssw=0.2
    # loss_path_train = [0.430, 0.344, 0.316, 0.272, 0.253, 0.223, 0.222, 0.216, 0.198, 0.182]
    # loss_path_eval = [0.224, 0.323, 0.353, 0.296, 0.276, 0.257, 0.214, 0.247, 0.220, 0.200]
    # loss_path_train_teacher = [0.202, 0.210, 0.192, 0.200, 0.189, 0.184, 0.176, 0.192, 0.218, 0.89]
    # loss_path_eval_teacher = [0.165, 0.175, 0.186, 0.165, 0.206, 0.162, 0.180, 0.164, 0.192, 0.171]

    # T_lr 8e-5
    # loss_path_train = [0.432, 0.428, 0.386, 0.366, 0.357, 0.338, 0.307, 0.298, 0.288, 0.272]
    # loss_path_eval = [0.271, 0.284, 0.295, 0.280, 0.252, 0.211, 0.227, 0.234, 0.224, 0.198]
    # loss_path_train_teacher = [0.539, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
    # loss_path_eval_teacher = [0.213, 0.800, 0.800, 0.800, 0.800, 0.800, 0.800, 0.800, 0.800, 0.800]

    plt.title('Loss Performance of SegFormer-Pseudo (Teacher-Student)')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim((0, 1))
    plt.plot(range(10), loss_path_train, color='red', label='student-train')
    # plt.plot(range(len(loss_path_eval)), loss_path_eval, color='orange', label='student-eval')
    # plt.plot(range(len(loss_path_train_teacher)), loss_path_train_teacher, color='green', label='teacher-train')
    # plt.plot(range(len(loss_path_eval_teacher)), loss_path_eval_teacher, color='blue', label='teacher-eval')
    plt.legend()
    plt.show()
