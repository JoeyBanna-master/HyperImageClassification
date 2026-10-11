#这里是一个epoch发生的训练
import time
import logging
import json
from pathlib import Path
from .val_function import validate
from .classification_metrics import ClassificationMetrics
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed time: %dh %dmin %ds' % (hour, minute, second))

# def train_epoch(args, train_loader, model, criterion, optimizer):
#     model.train()
#
#     # loss = 1
#     for step, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(args.device), targets.to(args.device)
#         optimizer.zero_grad()  #梯度归零
#         outputs = model(inputs)  #模型输出（分类情况）
#         loss = criterion(outputs, targets)  #根据真实类别和模型划分的类别情况计算损失
#         loss.backward()  #反向传播计算得到每个参数的梯度值
#         optimizer.step()  #通过梯度下降执行一步参数更新
#     # print(f"loss={loss.item():.4f}")

def train_epoch(args, train_loader, model, criterion, optimizer, epoch):
    model.train()
    statistics = ClassificationMetrics(args.classes)

    # loss = 1
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device, non_blocking=True), targets.to(args.device, non_blocking=True)
        optimizer.zero_grad()  #梯度归零
        outputs = model(inputs)  #模型输出（分类情况）
        loss = criterion(outputs, targets)  #根据真实类别和模型划分的类别情况计算损失
        loss.backward()  #反向传播计算得到每个参数的梯度值
        optimizer.step()  #通过梯度下降执行一步参数更新
        # Online metrics use each training batch's predictions BEFORE its update.
        statistics.update(outputs, targets, loss)
    # print(f"loss={loss.item():.4f}")
        # 只在第一个 step 打印 FLOPs 和显存信息
        if step == 0 and epoch ==299:
            # FLOPs
            flops = FlopCountAnalysis(model, inputs)
            total_flops = flops.total()
            print(f"[FLOPs] Epoch {epoch+1}: {total_flops / 1e9:.4f} GFLOPs")
            # 显存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(args.device) / (1024 ** 2)
                reserved = torch.cuda.memory_reserved(args.device) / (1024 ** 2)
                print(f"[GPU MEM] Epoch {epoch + 1} Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    return statistics.result()

#我们还需要一个总的训练函数，一次调用就代表我们实验重复一次
def train(args, iter, model, train_loader, val_loader, criterion, optimizer, scheduler, epoch_num,
          history_path=None):
    start = time.time()
    best_val_OA = 0.0   #测试集最好的总体分类精度
    best_model = None   #记录性能最好的模型的权重

    epoch_times = []
    training_history = []
    if history_path is not None:
        history_path = Path(history_path)
        history_path.write_text('', encoding='utf-8')
    for epoch in range(epoch_num):
        if args.device.type == 'cuda':
            torch.cuda.synchronize(args.device)
        epoch_start = time.perf_counter()
        # Choice Model Training
        print('-------Iter:{}....train weights for epoch:{}-------'.format(iter, epoch + 1))
        learning_rate = optimizer.param_groups[0]['lr']
        train_metrics = train_epoch(args, train_loader, model, criterion, optimizer, epoch)
        #更新学习率计划
        scheduler.step()
        # Choice Model Validation 验证模型
        if args.device.type == 'cuda':
            torch.cuda.synchronize(args.device)
        train_end = time.perf_counter()
        val_metrics = validate(model, val_loader, criterion=criterion, num_classes=args.classes,
                               return_metrics=True)
        val_OA = round(val_metrics['accuracy'] * 100, 2)  #保留原验证选择逻辑
        epoch_times.append({'epoch': epoch + 1, 'train_seconds': train_end - epoch_start,
                            'validation_seconds': time.perf_counter() - train_end})
        record = {
            'epoch': epoch + 1, 'learning_rate': learning_rate,
            'train_loss': train_metrics['loss'], 'train_accuracy': train_metrics['accuracy'],
            'train_samples': train_metrics['num_samples'],
            'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy'],
            'val_samples': val_metrics['num_samples'],
        }
        training_history.append(record)
        print(f"Epoch {epoch + 1}: train_loss={record['train_loss']:.6f}, "
              f"train_accuracy={record['train_accuracy'] * 100:.2f}%, "
              f"val_loss={record['val_loss']:.6f}, val_OA={val_OA:.2f}%, "
              f"lr={learning_rate:.6g}", flush=True)
        if history_path is not None:
            detailed_record = dict(
                record, train_confusion_matrix=train_metrics['confusion_matrix'],
                val_confusion_matrix=val_metrics['confusion_matrix'])
            with history_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(detailed_record, ensure_ascii=False) + '\n')
        # Save Best Model Weights  保存最佳模型权重参数
        if best_val_OA <= val_OA:
            best_val_OA = val_OA
            best_model = model
            print('Best val_OA by now,update the best model')

    print('*****Iter:{}....This iteration has ended, best_val_OA:{}, , return the best model*****'.format(iter, best_val_OA))
    # Record Time
    time_record(start)
    #这里只需要返回我们训练的最好权重即可
    args.epoch_times = epoch_times
    args.training_history = training_history
    return best_model
