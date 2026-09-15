# HSI 项目说明

## 技术栈

- Python / PyTorch 高光谱 patch 分类实验。
- 主要依赖：NumPy、scikit-learn、einops、torch-dct、fvcore。
- requirements.txt 中 PyTorch 版本被注释，Python 版本未声明。

## 项目结构

- Whole_FBCANet.py：实验入口。
- Config_for_datasets/Config.py：数据路径、波段数和类别数。
- Mydataset/High_Dataset.py：npy 样本与标签加载。
- model/FBCANet.py：特征分支、融合、分类器。
- functions/train_function_new.py：当前入口实际调用的训练实现。
- functions/val_function.py、test_function.py：模型评估。
- datasets/：原始数据及预处理脚本。

## 运行与验证

- 从项目根目录运行 python Whole_FBCANet.py；默认 10 次实验，每次 200 轮。
- 运行前核对配置对应的六个 npy 文件；2026-09-15 检查时默认 Houston 文件缺失。
- 未发现统一自动化测试、构建或 lint 命令；不要把模型测试函数当作单元测试。
- 修改模型时关注输入 [B,C,H,W]、输出 [B,num_classes] 以及分支内部的 NCHW/NHWC 转换。
- 本轮 onboarding 只做静态阅读，未验证训练可以运行。

## 代码约定与解释依据

- 函数主要使用 snake_case，网络使用 nn.Module 类，参数由 argparse 和配置字典传递。
- 模型文件包含许多注释掉的历史实验；跟踪实际构造与 forward，不把注释中的性能声明当作验证结果。
- 未检测到 Git 仓库，提交和分支约定未知。

## 已知问题

- SGD 位置参数使 weight_decay 值落入 dampening。
- best_model = model 未保存最佳轮次权重快照。
- 验证和测试 shuffle/drop_last 影响评估样本覆盖。
- validate 硬编码 CUDA，主入口的 CPU 回退不完整。
- 数据类别数、模型类别数和 patch 大小需要对齐；seed 当前没有生效。
- 最终模型由测试 OA 选择，解释评估结果时需说明。

更详细的流程和静态检查结果见 ONBOARDING.md。
