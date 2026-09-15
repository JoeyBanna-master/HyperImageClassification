
import argparse
import logging
import os
import sys

import numpy as np

import torch
import torch.nn as nn

from model.FBCANet import FBCANet

from fvcore.nn import parameter_count_table
from Config_for_datasets.Config import CONFIG_HSIC
# 导入我自己制作的datasloader
from Mydataset.High_Dataset import MyDataset
from torch.utils.data import DataLoader
from functions.val_function import validate
from functions.train_function_new import train
from functions.test_function import test

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
parser = argparse.ArgumentParser("Whole_FBCANet")
parser.add_argument('--exp_name', type=str, default='Whole_FBCANet', help='experiment name')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='checkpoints direction')
# Training Setting
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--iters', type=int, default=10, help='number of experiment repeats')
parser.add_argument('--learning_rate', type=float, default=0.005, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--seed', type=int, default=0, help='training seed')
parser.add_argument('--step_size', type=int, default=10, help='step_size')
parser.add_argument('--gamma', type=float, default=0.4, help='gamma')

parser.add_argument('--data_root', type=str, default='./datasets/', help='datasets dir')
parser.add_argument('--classes', type=int, default=15, help='datasets classes')
parser.add_argument('--datasets', type=str, default='HUST2013', help='path to the datasets')
parser.add_argument('--ablation', type=str, default='1', help='ablation')
parser.add_argument('--windows', type=int, default=11, help='size of patches')
parser.add_argument('--cutout', action='store_true', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')


args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)



# 固定随机因子，暂且留下，用时方便
# utils.set_seed(args.seed)

def get_model():
    model = FBCANet(in_channels=CONFIG_HSIC[args.datasets]['bands'], num_classes=args.classes, dim=64, reduction=16, patch_size=args.windows)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - (epoch / args.epochs))
    return model, optimizer, scheduler


# 训练选中的bestpath
def main():
    train_dataset = MyDataset(CONFIG_HSIC[args.datasets]['train_data_path'],
                              CONFIG_HSIC[args.datasets]['train_label_path'])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    val_dataset = MyDataset(CONFIG_HSIC[args.datasets]['val_data_path'],
                            CONFIG_HSIC[args.datasets]['val_label_path'])
    # 加载验证集
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True, drop_last=True)

    test_dataset = MyDataset(CONFIG_HSIC[args.datasets]['test_data_path'],
                             CONFIG_HSIC[args.datasets]['test_label_path'])
    # 加载验证集
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True, drop_last=True)
    print(CONFIG_HSIC[args.datasets]['bands'])
    print(CONFIG_HSIC[args.datasets]['num_classes'])


    criterion = nn.CrossEntropyLoss().to(args.device)

    # 保存最终的模型
    final_model = None
    final_OA = 0
    final_AA = 0
    final_kappa = 0
    final_report = 0
    eva_time = 0
    # 创建一个列表来存储各个迭代模型的OA
    OA_list = []
    AA_list = []
    Kappa_list = []
    OA_AA_Kappa_list = []
    Class_Acc = np.zeros((args.iters, CONFIG_HSIC[args.datasets]['num_classes']))

    # 接下来进入迭代的大循环
    for index_iter in range(args.iters):
        print('iter:', index_iter)
        # 每次迭代开始之前重置模型参数以及优化器等，这里用了个笨办法，重新申请了模型
        model, optimizer, scheduler = get_model()
        model = model.to(args.device)

        # 训练模型
        iter_model = train(args, index_iter, model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs)
        # 测试模型
        iter_OA, iter_AA, iter_kappa, iter_report, acc_for_each_class, time = test(args, index_iter, test_loader, iter_model)

        eva_time += time
        OA_list.append(iter_OA)
        AA_list.append(iter_AA)
        Kappa_list.append(iter_kappa)
        OA_AA_Kappa_list.append([iter_OA, iter_AA, iter_kappa])

        Class_Acc[index_iter, :] = acc_for_each_class

        # 保存最优迭代模型
        if final_OA < iter_OA:
            final_OA = iter_OA
            print("Best OA by now")
            final_model = iter_model  # 注意要保存当次迭代模型
            final_AA = iter_AA
            final_kappa = iter_kappa
            final_report = iter_report

    # 统计模型参数量
    print('======================================================')
    num_params = sum(param.numel() for param in final_model.parameters())
    print("模型参数量：")
    print(num_params)
    print("模型参数量核算：")
    print(parameter_count_table(final_model))
    print("模型infer速度：")
    print(eva_time / args.iters)

    print("================Final 整体测试集上的报告=================")
    print(final_report)
    print(OA_list)
    # 计算每个类别的平均
    mean_accuracies = np.mean(Class_Acc, axis=0)
    # 计算每个类别的标准差
    std_deviations = np.std(Class_Acc, axis=0, ddof=1)  # 使用样本标准差
    # 以平均准确率±标准差的形式输出
    for i in range(CONFIG_HSIC[args.datasets]['num_classes']):
        print(f"类别{i + 1} : {mean_accuracies[i] * 100:.2f} ± {std_deviations[i] * 100:.2f}")

    OA_AA_Kappa_list = np.array(OA_AA_Kappa_list)
    # 计算每个类别的平均
    mean_accuracies = np.mean(OA_AA_Kappa_list, axis=0)
    # 计算每个类别的标准差
    std_deviations = np.std(OA_AA_Kappa_list, axis=0, ddof=1)  # 使用样本标准差
    # 以平均准确率±标准差的形式输出
    name_list = ['OA', 'AA', 'Kappa']
    for i in range(3):
        print(f"{name_list[i]} : {mean_accuracies[i] * 100:.2f} ± {std_deviations[i] * 100:.2f}")

    # 保存最终模型
    save_path = 'model_pth/' + args.datasets + '/' + args.exp_name + '_' + 'ab' + '{' + str(
        args.ablation) + '}' + args.datasets + '.pth'
    if os.path.isdir('model_pth/' + args.datasets) is False:
        os.makedirs('model_pth/' + args.datasets)
    if os.path.isfile(save_path) is False:
        open(save_path, 'wt')
    torch.save(final_model, save_path)
    print('模型已保存')


if __name__ == '__main__':
    main()