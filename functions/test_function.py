'''
这里保存test代码，每次迭代调用一次测试代码
'''
import time
import sys
import torch
import numpy as np
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from torch.autograd import Variable

def test_mcmd(args, mapping_tar, encoder, train_loader, test_loader):
    """
    专门针对 MCMD 框架的测试函数
    机制：使用 mapping_tar 和 encoder 提取目标域标记训练集特征，训练 KNN 分类器；
          然后对测试集提取特征，用 KNN 进行预测。
    """
    mapping_tar.eval()
    encoder.eval()

    with torch.no_grad():
        # 1. 提取训练集特征来 Fit KNN 分类器
        train_datas, train_labels, index = next(train_loader.__iter__())
        train_features = encoder(mapping_tar(Variable(train_datas.float()).to(args.device)))

        # 特征归一化
        max_value = train_features.max()
        min_value = train_features.min()
        train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

        # 训练 KNN
        KNN_classifier = KNeighborsClassifier(n_neighbors=1)
        KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels.numpy())

        # 2. 遍历测试集进行推断
        predict_all = []
        labels_all = []

        for test_datas, test_labels, test_index in test_loader:
            test_features = encoder(mapping_tar((Variable(test_datas.float()).to(args.device))))
            test_features = (test_features - min_value) * 1.0 / (max_value - min_value)

            predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())

            predict_all.extend(predict_labels)
            labels_all.extend(test_labels.numpy())

    # 计算各项指标
    labels_all = np.array(labels_all)
    predict_all = np.array(predict_all)

    report = metrics.classification_report(labels_all, predict_all, digits=4)
    acc_for_each_class = metrics.precision_score(labels_all, predict_all, average=None)

    test_OA = metrics.accuracy_score(labels_all, predict_all)
    test_AA = np.mean(acc_for_each_class)
    test_kappa = metrics.cohen_kappa_score(labels_all, predict_all)

    return test_OA, test_AA, test_kappa, report, acc_for_each_class

def test(args, iter, test_loader, model):
    model.eval()
    # 创建一个predictions来记录我们预测的结果
    predictions = torch.FloatTensor().to(args.device)
    # labels是原本的标签
    labels = torch.FloatTensor().to(args.device)
    if args.device.type == 'cuda':
        torch.cuda.synchronize()
    T1 = time.perf_counter()  # 测试开始的时间
    with torch.no_grad():  # 是一个上下文管理器，被该语句 wrap 起来的部分将不会track梯度。
        for (patches, targets) in test_loader:
            patches = patches.to(args.device)
            targets = targets.to(args.device)
            output = model(patches)
            batch_prediction = output.argmax(1)
            # output,_,_,= model(patches) #DBMGNet专用
            # output, abu, hsi_re = model(patches)  # MSUANet
            # _, _, logits = model(patches, patches)  # LRDTN专用
            # # 记录这一个batch的预测结果
            # batch_prediction = logits.argmax(1) # LRDTN专用


            predictions = torch.cat((predictions, batch_prediction), 0)
            labels = torch.cat((labels, targets), 0)
    if args.device.type == 'cuda':
        torch.cuda.synchronize()
    T2 = time.perf_counter()  # 测试结束的时间
    print('程序推理时间:%s秒' % ((T2 - T1)))
    tim = (T2 - T1)

    #开始计算各个结果参数
    predictions = predictions.cpu()
    labels = labels.cpu()
    print("---Iter:{} 整体测试集上的报告---".format(iter))
    report = metrics.classification_report(labels, predictions, digits=4)
    print(report)
    # HSI 中的 AA 是各类别召回率（类别准确率）的平均，不是 precision。
    acc_for_each_class = metrics.recall_score(
        labels, predictions, average=None, zero_division=0
    )
    # OA得单独拎出来，因为要计算最优模型
    test_OA = metrics.accuracy_score(labels, predictions)
    print("整体测试集上的OA:{}".format(test_OA))
    test_AA = np.mean(acc_for_each_class)
    print("整体测试集上的AA:{}".format(test_AA))
    test_kappa = metrics.cohen_kappa_score(labels, predictions)
    print("整体测试集上的kappa系数:{}".format(test_kappa))
    print("test_OA={}".format(round(test_OA, 4)))

    return test_OA, test_AA, test_kappa, report, acc_for_each_class, tim


# import time
# import torch
# import numpy as np
# from sklearn import metrics
#
#
# def test(args, iter, test_loader, model):
#     model.eval()
#     # 初始化 predictions 和 labels
#     predictions = torch.IntTensor().to(args.device)
#     labels = torch.IntTensor().to(args.device)
#
#     T1 = time.time()
#     with torch.no_grad():
#         for (patches, targets) in test_loader:
#             patches = patches.to(args.device)
#             targets = targets.to(args.device)
#             output = model(patches)
#             batch_prediction = output.argmax(1)
#             predictions = torch.cat((predictions, batch_prediction), 0)
#             labels = torch.cat((labels, targets), 0)
#     T2 = time.time()
#
#     # 计算推理时间
#     tim = T2 - T1
#     print(f"程序推理时间: {tim:.2f} 秒")
#
#     # 计算性能指标
#     predictions = predictions.cpu()
#     labels = labels.cpu()
#
#     print(f"---Iter:{iter} 整体测试集上的报告---")
#     report = metrics.classification_report(labels, predictions, digits=4)
#     print(report)
#
#     acc_for_each_class = metrics.precision_score(labels, predictions, average=None)
#     test_OA = metrics.accuracy_score(labels, predictions)
#     test_AA = np.mean(acc_for_each_class)
#     test_kappa = metrics.cohen_kappa_score(labels, predictions)
#
#     print(f"整体测试集上的OA: {test_OA:.4f}")
#     print(f"整体测试集上的AA: {test_AA:.4f}")
#     print(f"整体测试集上的kappa系数: {test_kappa:.4f}")
#
#     # 返回测试指标
#     return test_OA, test_AA, test_kappa, report, acc_for_each_class, tim
