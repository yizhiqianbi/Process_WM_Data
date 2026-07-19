# Target 输出索引

[文档索引](../README.md)

本仓库区分两件事：

- **Source adapter**：读取 OXE、AgiBot、LingBot-VA data、DreamZero-DROID 等上游数据并审计语义。
- **Model target**：按某个模型训练代码的真实 loader 合同生成 metadata、统计和训练入口。

“能读取 LingBot-VA/DreamZero 数据”不等于“能为 LingBot-VA/DreamZero 模型生成训练数据”。当前三个 target 为：

| Target | 输入 | 输出 | 当前边界 |
|---|---|---|---|
| [FastWAM](FASTWAM.md) | 九类 source adapter | 81/80/21 `TrainingCaseV1` | 完整 canonical target 路径 |
| [old LingBot-VA](LINGBOT_VA.md) | LeRobot v2 | `action_config`、compact-to-30D contract、quantile stats、latent jobs | 提取全部 VAE latent 后正式训练；局部 latent 仅用于 smoke |
| [DreamZero](DREAMZERO.md) | LeRobot v2 | GEAR modality、stats、relative stats、Hydra profile | 安装器校验 embodiment/projector 并安装数据 profile |

Target 输出默认是非破坏 overlay：原始数据只读，metadata 写入新目录；大体积 `data/` 和 `videos/` 可使用 symlink、hardlink 或 copy。

## Readiness 规则

- `valid=true`：target metadata、维度、相机和数值合同通过。
- `ready_for_training=true`：额外的模型侧产物和注册条件也满足。
- LingBot-VA 缺少任一 VAE latent 时，必须保持 `ready_for_training=false`。
- DreamZero 自定义 embodiment 未合并 Hydra/profile patch 时，必须保持 `ready_for_training=false`。
- FastWAM 的样本级 case 通过不代表九库全量训练已完成。
