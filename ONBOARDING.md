# HSI / FBCANet 代码入门

基于 2026-09-15 本地源码静态阅读；未运行训练。目标项目：D:/Projects/高光谱/HSI。

## 项目与技术栈

这是基于高光谱图像 patch 的分类实验项目，入口为 Whole_FBCANet.py。使用 Python、PyTorch、NumPy、scikit-learn、einops、torch-dct、fvcore；Python 版本未声明，requirements.txt 中 torch==2.5.1 被注释，不能据此确定实际版本。

## 文件地图

| 文件或目录 | 作用 |
| --- | --- |
| Whole_FBCANet.py | 参数、数据加载、重复实验、结果汇总、模型保存 |
| Config_for_datasets/Config.py | 数据集波段数、类别数、预处理数据路径 |
| Mydataset/High_Dataset.py | 从成对的 npy 文件读取样本和标签 |
| functions/train_function_new.py | 实验内的 epoch 循环和 batch 参数更新 |
| functions/val_function.py | 验证集预测与 OA |
| functions/test_function.py | 测试集预测、OA、AA、Kappa 与计时 |
| model/FBCANet.py | 四条特征路径、融合与分类器 |
| datasets/ | 原始数据、预处理脚本、预期的 npy 数据 |
| 十篇对比方法/ | 对比方法代码，不是本入口的主调用链 |

## 从 Whole 开始的执行顺序

1. 第 2–52 行：导入依赖，解析命令行参数，选择设备，设置日志。
2. 第 59–63 行：定义 get_model；定义本身不会立即创建网络。
3. 第 178–179 行：直接运行文件时调用 main。
4. main 第 68–86 行：按 CONFIG_HSIC 读取训练、验证、测试数据，分别建立 DataLoader。
5. 第 91–105 行：建立交叉熵损失以及实验结果容器。
6. 第 108–134 行：重复 iters 次，每次重新创建网络、优化器、调度器，调用 train，再调用 test，记录指标；按测试 OA 选择最终模型。
7. 第 136–165 行：打印参数量、平均测试耗时、分类报告、各类和总体指标的均值及样本标准差。
8. 第 168–174 行：保存完整模型对象到 model_pth/<数据集>/ 下。

默认 iters=10、epochs=200、batch_size=32：每次实验训练 200 轮，总共重复 10 次；一轮内遍历训练批次。重复实验复用同一套数据集，没有重新划分数据。

## 单个 batch 的旅程

默认 HUST2013 配置为 144 个波段、15 类、11×11 patch。预期模型输入为 [B,144,11,11]，标签为 [B]，类别编号应为 0–14。数据文件目前缺失，因此这里是接口要求，不是实测数据形状。

MyDataset 使用 np.load 读取数组，并转换为 float32/int64；默认 DataLoader 将返回的 NumPy 样本组合成批量 Tensor。Dataset 中的逐样本 Tensor 赋值没有把整个 NumPy 容器变成 Tensor。

train_epoch：搬到设备 → 清梯度 → model(inputs) → 得到 [B,15] 类别分数 → 交叉熵 → backward → optimizer.step。

FBCANet.forward：

```text
输入 [B,C,H,W]
├─ SpectralLocal  ─┐
├─ SpectralGlobal ─┴─ 相加得到 combined_spectral ─┐
├─ SpatialLocal   ─┐                            ├─ DoubleBranchFusion → Classifier → [B,类别数]
└─ SpatialGlobal  ─┴─ 相加得到 combined_spatial ──┘
```

两路相加前统一为 [B,H,W,64]。Classifier 将其转为 NCHW，做全局平均池化、Flatten、Linear、ReLU、Dropout、Linear，输出未归一化分数。

每轮 train_epoch 结束，先 scheduler.step，再 validate。验证不反向传播，用 argmax 得到预测类别。验证 OA 返回百分数，test 的 OA/AA 返回比例；AA 为逐类 recall 的均值。测试结束后主入口统计各次实验的指标。

## 实际行为与易误读点

- train_function_new.py:71 的 best_model = model 只保存引用，后续训练继续改变同一对象，所以没有保留最佳验证轮次快照。
- Whole_FBCANet.py:61 的 SGD 第四个位置参数是 dampening；args.weight_decay 没有传给真正的 weight_decay。若修复，应使用显式关键字参数。参见 https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html 。
- 三个 DataLoader 都设置 shuffle=True、drop_last=True；验证和测试可能随机丢弃末尾不足一个批次的样本。
- 主入口支持选择 CPU，但 validate 内硬编码 .cuda()，CPU 路径不完整。
- --seed 对应的设置被注释；--step_size、--gamma 没有控制当前线性学习率调度；--data_root 没有参与实际数据路径；--ablation 只影响此入口的保存文件名。
- --classes 与 CONFIG_HSIC 的 num_classes 分别用于模型和统计，切换数据集时需要一致；--windows 要与预处理文件的 patch 大小一致。
- 最终模型按测试集 OA 选择，因此最终最佳报告是测试集参与选择后的结果，不宜当作独立、无偏的模型选择评估。
- 默认所需的 Houston 六个 npy 文件均不存在；目录里的 .npy.filepart 不能替代完整文件。
- iters=1 时 ddof=1 的样本标准差没有定义；iters=0 或所有实验 OA=0 时 final_model 可能仍为空。
- 打印的“infer速度”实际是平均每次测试循环的秒数，包含数据搬运等操作，不是每秒处理样本数。

## 运行与项目约定

应从 HSI 根目录运行，因为数据与输出使用相对路径。准备环境、补齐预处理数据并处理上述运行问题后，可用 python Whole_FBCANet.py 启动；此命令默认执行完整 10×200 轮实验。

代码采用脚本入口、nn.Module 类和独立训练/验证/测试函数，模型文件保留大量注释掉的实验实现；以实际实例化的类与 forward 为准。未发现统一测试/构建/格式检查配置；本目录不是 Git 仓库，无法判断提交及分支约定。functions/test_function.py 是模型评估代码，不是自动化测试套件。

推荐阅读顺序：Whole_FBCANet.py → Config.py → High_Dataset.py → train_function_new.py → FBCANet.forward → 各分支 → validate/test。
