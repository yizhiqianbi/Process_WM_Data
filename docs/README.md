# 文档索引

本目录是 `Process_WM_Data` 的统一文档入口。文档按 source data、清洗、model target、训练和状态拆分，避免同一个状态、命令或合同在多个文件中维护。

## 名词边界

- **LingBot-VA source data**：RoboTwin/LIBERO 等被 source adapter 读取的数据。
- **old LingBot-VA target**：`Robbyant/lingbot-va` 模型的 LeRobot/action_config/latent 训练合同。
- **LingBot-VLA v2**：`Robbyant/lingbot-vla-v2` 模型及独立微调仓库，不是本仓库的 dataset adapter。
- **FastWAM**：消费 `TrainingCaseV1` 的 world + action model；其模型代码和 checkpoint 不存放在本仓库。
- **DreamZero target**：`dreamzero0/dreamzero` 的 GEAR modality、relative stats 和 Hydra 合同。

这些模型可以共享原始机器人数据，但不能混用 normalization stats、action padding、视频输入布局或 checkpoint。

## Model Target

1. [Target 输出总览](targets/README.md)
2. [FastWAM](targets/FASTWAM.md)
3. [old LingBot-VA](targets/LINGBOT_VA.md)
4. [DreamZero](targets/DREAMZERO.md)

## 建议阅读路径

首次接入 FastWAM：

1. [当前验收状态](reference/VALIDATION_STATUS.md)
2. [执行路线图](ROADMAP.md)
3. [统一清洗管线](data/PREPROCESSING.md)
4. [Action 准入证据](data/ACTION_ADMISSION.md)
5. [FastWAM 数据接口](training/FASTWAM_DATA_INTERFACE.md)
6. [FastWAM 三阶段训练](training/FASTWAM_THREE_STAGE.md)
7. [三模型统一微调](training/THREE_MODEL_TUNING.md)
8. [FastWAM 天机单轨迹过拟合](training/FASTWAM_TIANJI_OVERFIT.md)

新增或排查某个数据源：

1. [下载与完整性验证](data/DOWNLOAD.md)
2. [Action 准入证据](data/ACTION_ADMISSION.md)
3. 对应的数据集专题文档
4. [统一清洗管线](data/PREPROCESSING.md)

## 文档职责

| 文档 | 唯一职责 |
|---|---|
| [Roadmap](ROADMAP.md) | 记录下一步执行顺序、阶段门槛和完成定义 |
| [Validation Status](reference/VALIDATION_STATUS.md) | 记录已经实际运行并获得证据的结果 |
| [Download](data/DOWNLOAD.md) | 数据下载、断点续传、revision 锁定和完整性检查 |
| [Preprocessing](data/PREPROCESSING.md) | 通用清洗算法、分级、坏区间、窗口准入和阈值 |
| [Action Admission](data/ACTION_ADMISSION.md) | 每个数据源的 action 语义、映射证据和降级原因 |
| [AgiBot](datasets/AGIBOT.md) | AgiBot HDF5 join、proprio mapping、统计和训练注意事项 |
| [LingBot-VA / DreamZero](datasets/LINGBOT_VA_DREAMZERO.md) | 两个扩展数据源的 schema、相机、action 和真实样本结果 |
| [Target Overview](targets/README.md) | source adapter 与模型 target 的边界和 readiness 规则 |
| [Old LingBot-VA Target](targets/LINGBOT_VA.md) | 30D channel、action_config、quantile stats 和 VAE latent jobs |
| [DreamZero Target](targets/DREAMZERO.md) | GEAR modality、relative stats、语言、Hydra profile 和安装校验 |
| [FastWAM Data Interface](training/FASTWAM_DATA_INTERFACE.md) | `TrainingCaseV1 -> batch -> loss` 的稳定接口合同 |
| [FastWAM Three-Stage](training/FASTWAM_THREE_STAGE.md) | Stage 1/2/3、memory、参数冻结和 checkpoint 交接 |
| [Three-Model Tuning](training/THREE_MODEL_TUNING.md) | 同一 source 到 FastWAM、old LingBot-VA、DreamZero 的启动、恢复和收据 |
| [FastWAM Tianji Overfit](training/FASTWAM_TIANJI_OVERFIT.md) | 固定窗口、memory-aware rollout、想象 vs GT demo 和验收门槛 |

## 单一事实来源

文档更新遵循以下规则：

- “已经跑通什么”只更新 `reference/VALIDATION_STATUS.md`。
- “接下来先做什么”只更新 `ROADMAP.md`。
- 下载命令和 revision 只维护在 `data/DOWNLOAD.md`。
- 清洗阈值和算法只维护在 `data/PREPROCESSING.md` 及版本化 YAML。
- 每库 action 语义和证据只维护在 `data/ACTION_ADMISSION.md` 或数据集专题文档。
- 模型专用输出合同只维护在 `targets/` 文档和 `configs/targets/`。
- FastWAM batch/loss 字段只维护在 `training/FASTWAM_DATA_INTERFACE.md`。
- 三阶段模型结构和 checkpoint 规则只维护在 `training/FASTWAM_THREE_STAGE.md`。
- 三模型统一 launcher 和跨仓库运行方式只维护在 `training/THREE_MODEL_TUNING.md`。
- 天机单轨迹过拟合样本、超参、demo 和验收门槛只维护在 `training/FASTWAM_TIANJI_OVERFIT.md`。

当实现与文档冲突时，版本化配置和测试是最终依据；文档必须在同一提交中修正。

## 状态用语

- **Implemented**：代码路径存在，但不表示已经用真实数据运行。
- **Sample-validated**：至少一个真实样本通过指定回归。
- **Dataset-validated**：固定 revision 的全量 manifest、文件和统计通过。
- **Training-validated**：完成 optimizer step、checkpoint 保存和 resume。
- **Production-ready**：全量数据、分布式训练、held-out 评测和可复现记录全部满足门槛。

文档中不得用“已调通”替代上述具体状态。
