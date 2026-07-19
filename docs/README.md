# 文档索引

本目录是 `Process_WM_Data` 的统一文档入口。文档按职责拆分，避免同一个状态、命令或合同在多个文件中维护。

## 名词边界

- **LingBot-VA**：本仓库处理的数据源，包括 RoboTwin 和 LIBERO 等上游数据。
- **LingBot-VLA v2**：`Robbyant/lingbot-vla-v2` 模型及独立微调仓库，不是本仓库的 dataset adapter。
- **FastWAM**：消费 `TrainingCaseV1` 的 world + action model；其模型代码和 checkpoint 不存放在本仓库。

三者可以共享原始机器人数据，但不能混用 normalization stats、action padding、视频输入布局或 checkpoint。

## 建议阅读路径

首次接入 FastWAM：

1. [当前验收状态](reference/VALIDATION_STATUS.md)
2. [执行路线图](ROADMAP.md)
3. [统一清洗管线](data/PREPROCESSING.md)
4. [Action 准入证据](data/ACTION_ADMISSION.md)
5. [FastWAM 数据接口](training/FASTWAM_DATA_INTERFACE.md)
6. [FastWAM 三阶段训练](training/FASTWAM_THREE_STAGE.md)

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
| [FastWAM Data Interface](training/FASTWAM_DATA_INTERFACE.md) | `TrainingCaseV1 -> batch -> loss` 的稳定接口合同 |
| [FastWAM Three-Stage](training/FASTWAM_THREE_STAGE.md) | Stage 1/2/3、memory、参数冻结和 checkpoint 交接 |

## 单一事实来源

文档更新遵循以下规则：

- “已经跑通什么”只更新 `reference/VALIDATION_STATUS.md`。
- “接下来先做什么”只更新 `ROADMAP.md`。
- 下载命令和 revision 只维护在 `data/DOWNLOAD.md`。
- 清洗阈值和算法只维护在 `data/PREPROCESSING.md` 及版本化 YAML。
- 每库 action 语义和证据只维护在 `data/ACTION_ADMISSION.md` 或数据集专题文档。
- FastWAM batch/loss 字段只维护在 `training/FASTWAM_DATA_INTERFACE.md`。
- 三阶段模型结构和 checkpoint 规则只维护在 `training/FASTWAM_THREE_STAGE.md`。

当实现与文档冲突时，版本化配置和测试是最终依据；文档必须在同一提交中修正。

## 状态用语

- **Implemented**：代码路径存在，但不表示已经用真实数据运行。
- **Sample-validated**：至少一个真实样本通过指定回归。
- **Dataset-validated**：固定 revision 的全量 manifest、文件和统计通过。
- **Training-validated**：完成 optimizer step、checkpoint 保存和 resume。
- **Production-ready**：全量数据、分布式训练、held-out 评测和可复现记录全部满足门槛。

文档中不得用“已调通”替代上述具体状态。
