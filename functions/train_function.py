#这里是一个epoch发生的训练
import time
import logging
from .val_function import validate
import torch.nn.functional as F
import psutil
import os

import torch
import torch.nn as nn
# from independent.MCMD_CFSL_2025.utils import loss_function
from sklearn import metrics
# from independent.MCMD_CFSL_2025.utils.dataloader import Task, get_HBKC_data_loader
from .test_function import test_mcmd # 引入专属测试函数
def train_mcmd(args, iter_index, models, train_opt, train_loader, test_loader, test_loader2, bert_feats, metatrain_data,
               target_da_metatrain_data):
    # 展开模型
    mapping_src = models['mapping_src']
    mapping_tar = models['mapping_tar']
    encoder = models['encoder']
    Siamese_model = models['siamese']
    mapping_tar_kuai = models['mapping_tar_kuai']
    mapping_src_kuai = models['mapping_src_kuai']

    # 展开BERT特征
    w1_src, w2_tar, g1_src, g2_tar = bert_feats

    # 优化器设置
    optim_src = torch.optim.SGD(mapping_src.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=train_opt['weight_decay'])
    optim_tar = torch.optim.SGD(mapping_tar.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=train_opt['weight_decay'])
    optim_enc = torch.optim.SGD(encoder.parameters(), lr=args.lr, momentum=0.9, weight_decay=train_opt['weight_decay'])
    # ... (省略 mapping_tar_kuai 等的优化器)

    crossEntropy = nn.CrossEntropyLoss().to(args.device)
    infoNCE_Loss = loss_function.ContrastiveLoss(batch_size=args.tar_class_num).to(args.device)

    best_test_OA = 0.0
    best_results = None

    for episode in range(args.episodes):
        mapping_src.train();
        mapping_tar.train();
        encoder.train()

        # 1. 构建本次回合的 Task (支持集和查询集)
        task_src = Task(metatrain_data, args.tar_class_num, train_opt['shot_num_per_class'],
                        train_opt['query_num_per_class'])
        support_dataloader_src = get_HBKC_data_loader(task_src, num_per_class=train_opt['shot_num_per_class'],
                                                      split="train", shuffle=False)
        # ... (此处补全目标域任务构建和 DataLoader 数据获取，参考原版 for episode 循环头部代码) ...

        # 2. 特征提取与前向传播
        # encoder(...) -> 获得 features, spatial, spectral 等
        # Siamese_model(...) -> 获得代理特征

        # 3. 损失计算
        # 计算欧式距离度量 f_loss, 计算蒸馏损失 z_loss, 文本对齐损失 text_align_loss
        # loss = f_loss + 2 * (text_align_loss_src + text_align_loss_tar) + (z_loss_t + z_loss_t_k + z_loss_s + z_loss_s_k)

        # 4. 反向传播更新
        # optim.zero_grad() -> loss.backward() -> optim.step()
        # 动量更新 Siamese 参数

        # 每隔一定轮次 (例如 500) 进行一次测试 (MCMD使用的是提取特征+KNN)
        if (episode + 1) % 500 == 0 or episode == 0:
            mapping_tar.eval()
            encoder.eval()

            test_OA, test_AA, test_Kappa, report, class_acc = test_mcmd(
                args, mapping_tar, encoder, train_loader, test_loader
            )

            if test_OA > best_test_OA:
                best_test_OA = test_OA
                best_results = (test_OA, test_AA, test_Kappa, class_acc, report, encoder, mapping_tar)
                print(f"Iter {iter_index} - Episode {episode + 1} - New Best OA: {best_test_OA:.4f}")

    return best_results

def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed time: %dh %dmin %ds' % (hour, minute, second))

def kl_divergence_loss(out1, out2):

    out1_log = F.log_softmax(out1, dim=-1)
    out2_soft = F.softmax(out2, dim=-1)
    return F.kl_div(out1_log, out2_soft, reduction='batchmean')

def train_epoch(args, train_loader, model, criterion, optimizer):
    model.train()
    loss = 1
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()  #梯度归零
        outputs = model(inputs)  #模型输出（分类情况）
        # outputs, abu, hsi_re = model(inputs) #MSUANet

        # print("Logits mean/std:", outputs.mean().item(), outputs.std().item())
        # _, _, logits = model(inputs, inputs)#LRDTN
        # loss = criterion(logits, targets)  # LRDTN
        # 热修复：如果模型吐出了 4D 空间特征图，使用全局平均池化拍扁它
        if outputs.dim() == 4:
            outputs = outputs.mean(dim=(2, 3))
        # 预防是 3D 序列输出 (Mamba 常见输出如 [32, 121, 15])
        elif outputs.dim() == 3:
            outputs = outputs.mean(dim=1)

        # 此时 outputs 一定是被规整过的标准二维矩阵 [32, 15] 了
        loss = criterion(outputs, targets)
        # loss = criterion(outputs.squeeze(), targets)  #根据真实类别和模型划分的类别情况计算损失
        # print("Loss:", loss.item())
        loss.backward()  #反向传播计算得到每个参数的梯度值
        optimizer.step()  #通过梯度下降执行一步参数更新
    # print(f"loss={loss.item():.4f}")

    # # DBMGNet模型专用
    # for step, (inputs, targets) in enumerate(train_loader):
    #     inputs, targets = inputs.to(args.device), targets.to(args.device)
    #     optimizer.zero_grad()  #梯度归零
    #     outputs, out1, out2 = model(inputs)  #模型输出（分类情况）
    #     loss = criterion(outputs, targets) + criterion(out1, targets) + criterion(out2, targets) + kl_divergence_loss(out1, out2) + kl_divergence_loss(out2, out1)  #根据真实类别和模型划分的类别情况计算损失
    #     loss.backward()  #反向传播计算得到每个参数的梯度值
    #     optimizer.step()  #通过梯度下降执行一步参数更新


#我们还需要一个总的训练函数，一次调用就代表我们实验重复一次
def train(args, iter, model, train_loader, val_loader, criterion, optimizer, scheduler, epoch_num):
    start = time.time()
    best_val_OA = 0.0   #测试集最好的总体分类精度
    best_state = None   # 将最佳权重复制到 CPU，避免只保存会继续变化的模型引用

    # 记录初始内存
    # process = psutil.Process(os.getpid())
    # initial_mem = process.memory_info().rss / 1024 ** 2  # MB

    for epoch in range(epoch_num):
        # Choice Model Training
        print('-------Iter:{}....train weights for epoch:{}-------'.format(iter, epoch + 1))
        train_epoch(args, train_loader, model, criterion, optimizer)
        #更新学习率计划
        scheduler.step()
        # Choice Model Validation 验证模型
        val_OA = validate(model, val_loader)  #验证集的总体分类精度

        print('val weights for epoch:{},val_OA={}'.format(epoch + 1, val_OA))

        # 打印当前内存
        # current_mem = process.memory_info().rss / 1024 ** 2
        # print(f"[Memory] Current RAM Usage: {current_mem:.2f} MB | Delta: {current_mem - initial_mem:.2f} MB")


        # Save Best Model Weights  保存最佳模型权重参数
        if best_val_OA <= val_OA:
            best_val_OA = val_OA
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            print('Best val_OA by now,update the best model')

    # 输出最终内存
    # peak_mem = process.memory_info().rss / 1024 ** 2
    # print(f"\n[Final Memory] Initial: {initial_mem:.2f} MB | Peak: {peak_mem:.2f} MB | Total Used: {peak_mem - initial_mem:.2f} MB")

    print('*****Iter:{}....This iteration has ended, best_val_OA:{}, , return the best model*****'.format(iter, best_val_OA))
    # Record Time
    time_record(start)
    # 恢复真正的验证集最优权重；原来的 best_model=model 只保存了可变引用。
    if best_state is None:
        raise RuntimeError("No validation checkpoint was recorded during training")
    model.load_state_dict(best_state)
    return model
