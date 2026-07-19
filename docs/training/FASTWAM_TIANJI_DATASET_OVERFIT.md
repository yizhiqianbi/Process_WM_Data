# FastWAM 天机全采集数据去鱼眼过拟合

[文档索引](../README.md)

更新日期：2026-07-19

本文定义 `take_wrong_item_right_arm` 的新一轮 FastWAM memorization 实验：先把三路训练画面从
鱼眼投影转换为统一的虚拟针孔视角，再让 44 条采集轨迹共同参与训练，并同时输出想象视频、
真实执行视频和动作曲线。它替代“只在一个 81 帧窗口上训练”的正式目标，但保留单窗口实验作为
训练链路诊断，见 [FastWAM 天机单窗口过拟合](FASTWAM_TIANJI_OVERFIT.md)。

## 1. 旧实验与新实验的边界

旧实验实际使用：

- 一个 episode 中的一个 4 秒窗口，dataset index 0（原始 `sample_offset=1` 后）。
- 原始鱼眼画面直接 resize 后拼成 `[3,21,384,320]`。
- 累计 900 steps 后，在同一个冻结窗口达到 26.87 dB、0.931 SSIM、action L1 0.0194。

该结果只证明模型、memory、action mask、checkpoint 和 rollout 链路能记住一个样本。它没有
证明模型见过其余 43 条轨迹，也不能说明模型学会了任务分布。

新实验使用：

| 项目 | 值 |
|---|---:|
| source episodes | 44 |
| source frames | 31,359 @ 28 Hz |
| canonical cases | 127 |
| total 4-second windows | 486 |
| joint video + action windows | 381 |
| video-only windows | 105 |
| selected cameras | 1 global + 2 wrist |
| model video | `[3,21,384,320]` |
| action / proprio | `[80,80]` / `[80,80]`，8 个有效维 |

105 个 video-only 窗口来自动作/state 坏区间切分。它们继续训练 video loss，但
`action_loss_mask=false`、动作维全部 padding，不会把异常动作写进 ActionDiT。

## 2. 相机内容绑定

采集时四路数据线与名字不一致，训练始终按画面内容绑定：

| 原始 source key | 实际内容 | 训练角色 |
|---|---|---|
| `observation.images.left_eye` | 主全局/头部视角 | `global_primary` |
| `observation.images.right_eye` | 左腕视角 | `left_wrist` |
| `observation.images.right_wrist` | 右腕视角 | `right_wrist` |
| `observation.images.left_wrist` | 第二全局/辅助视角 | `auxiliary`，当前不进 composite |

部署时也必须按该表绑定物理画面，不能仅根据 key 名推断相机位置。

## 3. 鱼眼到虚拟针孔

### 3.1 当前证据边界

源数据只有 960x744 视频，没有厂家内参或标定板结果；manifest 明确记录
`intrinsics_available=false`。因此当前配置是经过真实画面检查的**近似投影**，不是测量标定：

```text
source model: OpenCV fisheye / equidistant
source image: 744 x 960
fx = fy = 960 / pi = 305.5774907364
cx = 479.5, cy = 371.5
D = [0, 0, 0, 0]
assumed full horizontal fisheye FOV = 180 degrees
virtual pinhole horizontal FOV = 110 degrees
```

版本化配置为：

```text
configs/cameras/tianji_xdof_fisheye_approx_v1.json
SHA256 58bd5d2dcf471e33f95243a32577ae871081669c6af288c43c383ba95eea765d
```

该参数能把货架立柱、门框等直线恢复为近似直线，并保留主相机中的双臂和操作区域。真实部署前
仍应对四个物理相机分别做棋盘格或 AprilTag 标定，用测量的 K/D 替换 profile；接口和训练命令
无需变化。

### 3.2 loader 顺序

每路画面使用如下顺序：

```text
exact source frame decode
  -> fisheye-to-pinhole grid_sample
  -> role-specific output size
  -> three-camera composite
  -> [-1, 1]
```

虚拟投影直接生成最终 panel 分辨率，避免先生成 960x744 中间视频：

```text
global:      256 x 320
left wrist:  128 x 160
right wrist: 128 x 160
composite:   384 x 320
```

同一个 rectification 同时作用于 21 个目标帧和 8/2/1 memory 帧，不会出现历史画面仍是鱼眼、
未来画面已经去畸变的域错位。profile 未绑定的其他数据集保持原 resize；已绑定相机若输入尺寸
不是 744x960 则立即报错，不会静默拉伸。

### 3.3 预览与收据

```bash
/path/to/fastwam/python scripts/preview_fastwam_rectification.py \
  --config work/tuning/take_wrong_item.local.yaml \
  --phase dataset_overfit \
  --sample-index 5 \
  --output work/tuning/previews/tianji_sample_000005_fisheye_vs_pinhole.png
```

同名 JSON 记录 case、episode、window start、profile SHA 和三路 applied mask。当前真实样本结果：

```text
camera_present_mask               = [true, true, true]
camera_rectification_applied_mask = [true, true, true]
case_id = f0760af5c9dfd420f5af1854
window_start = 200
```

## 4. 全数据展开与采样

`dataset_overfit` 显式设置：

```yaml
allowed_modes: [joint_video_action, video_only]
splits: [train, validation]
sample_offset: 0
max_samples: null
sampling_strategy: uniform
```

`sample_offset: 0` 很重要。所复用的 upstream smoke task 默认带 `sample_offset=1`；若不覆盖，
loader 只得到 485 个窗口。真实 validation 已确认当前命令得到 486 个窗口。

这里把 43 个 train episode 和 1 个原 validation episode 都加入训练，是因为目标明确是“在当前
全部采集数据上 memorization/overfit”。因此本实验没有 held-out 指标，任何结果都不能当作泛化
性能。后续正式策略微调必须重新按 episode 划分 train/validation/test。

`uniform` 每个 epoch 对 486 个窗口做一次无放回乱序，保证每个窗口都被消费。这里不做
episode-balanced oversampling，因为那会在单个 epoch 中重复短轨迹并漏掉长轨迹的部分窗口；
当前目标首先是完整覆盖所有采集窗口。

## 5. 跨 episode 固定 probe

生成与 loader 完全相同顺序的 probe 计划：

```bash
python3 scripts/plan_fastwam_dataset_overfit.py \
  --manifest work/tuning/fastwam_data/lingbot_va/cases/training_cases.jsonl \
  --output work/tuning/fastwam_data/dataset_overfit_plan.json \
  --training-probe-count 8
```

选择规则：每条 episode 优先选 `joint_video_action`，再优先 `window_start >= 40` 的完整 memory
窗口，最后选该 episode 的中间窗口。完整计划包含 44 个一轨迹一 probe；训练期间固定使用均匀
覆盖 episode 0/6/12/18/25/31/37/43 的 8 个 dataset indices：

```text
[5, 87, 180, 239, 318, 377, 441, 480]
```

每次评测会在同一个 checkpoint 上运行整组 8 个 probe。单样本 JSON/视频文件名包含
`eval_<index>`，suite JSON 报告 8 条轨迹的均值并保留逐 probe 记录。因此 step 0 与后续 step
可以逐样本比较，不会把不同轨迹的 PSNR 或 action L1 错当成同一曲线。

## 6. 训练配置

`configs/tuning/take_wrong_item.example.yaml` 中的 `dataset_overfit` 默认：

| 配置 | 值 |
|---|---:|
| initialization | Stage-2 MemoryFastWAM checkpoint |
| batch / accumulation | 1 / 1 |
| backbone LR | `2e-5` |
| memory/reference LR | `1e-4` |
| video/action lambda | 1 / 1 |
| weight decay | 0 |
| scheduler | 5% linear warmup, then constant |
| sampler | uniform, 486 samples/epoch |
| workers | 2 |
| checkpoint interval | 500 steps |
| 8-probe eval interval | 2,000 steps + step 0 |
| inference steps / seed | 10 / 42 |

10,000 optimizer steps 约等于 20.6 个 dataset epochs。前 500 steps 将 LR 从接近 0 线性升到
配置值，之后保持 constant。先做 1 到 2 step smoke，再根据 2,000、
4,000 step 的 8-probe 曲线决定是否跑满 10,000；不能只看某一条 episode 的最好视频。

## 7. 启动顺序

先确认命令中的关键字段为 486 全量语义：

```bash
python3 scripts/tune_models.py dry-run \
  --config work/tuning/take_wrong_item.local.yaml \
  --model fastwam \
  --phase dataset_overfit \
  --output-dir work/tuning/runs/fastwam_tianji_dataset_overfit \
  --steps 10000 \
  --gpus 0
```

必须看到：

```text
data.train.max_samples=null
++data.train.sample_offset=0
data.train.split=[train,validation]
data.train.allowed_modes=[joint_video_action,video_only]
++data.train.camera_rectification_config=...tianji_xdof_fisheye_approx_v1.json
++eval_fixed_indices=[5,87,180,239,318,377,441,480]
```

真实两步 smoke：

```bash
python3 scripts/tune_models.py run \
  --config work/tuning/take_wrong_item.local.yaml \
  --model fastwam \
  --phase dataset_overfit \
  --output-dir work/tuning/runs/fastwam_tianji_dataset_overfit_smoke \
  --steps 2 \
  --gpus 0
```

smoke 通过后再启动正式训练：

```bash
python3 scripts/tune_models.py run \
  --config work/tuning/take_wrong_item.local.yaml \
  --model fastwam \
  --phase dataset_overfit \
  --output-dir work/tuning/runs/fastwam_tianji_dataset_overfit \
  --steps 10000 \
  --gpus 0
```

## 8. 视频与动作产物

每个固定 probe 都保留原三联诊断视频，并新增：

```text
*_imagination.mp4
*_execution.mp4
*_action.mp4
*_imagination_vs_execution_action.mp4
*_actions.npz
*.json
```

- `imagination`：FastWAM 生成的 21 帧未来。
- `execution`：同一首帧、同一时间窗的真实采集视频。
- `action`：80 个 20 Hz action step；GT 为橙色，模型预测为青色，白色游标与 21 帧视频同步。
- 合成视频：上方左右为 imagination / GT execution，下方为 8 个有效 slot 的动作曲线。
- NPZ：同时保存 normalized 与反归一化 action、GT、预测及有效 mask，便于数值分析。

动作图只画 slot 13、21..27。其余 72 个 canonical padding 维在训练和 flow-matching 推理的每一步
都强制为零，不会出现在图中。

当前视频 expert 仍是 `action_conditioned=false`。因此 imagination 与 action 是同一模型共同预测
的两个输出，不应描述为“给定这条 GT action 后生成的视频”。如果要比较两条候选动作导致的
反事实未来，需要后续单独训练 action-conditioned video expert。

## 9. 验收门槛

数据与投影门槛：

1. loader 报告 127 cases、486 windows。
2. 三路 `camera_rectification_applied_mask` 均为 true。
3. rectification profile SHA 与训练收据一致。
4. 381 个 joint 窗口有 8 个有效 action/state slots；105 个 video-only 窗口 action loss 为零。

训练门槛：

1. step-0 与后续 suite 使用相同 8 个 indices、相同 seed 和 inference steps。
2. 8 条 probe 的 PSNR/SSIM 均值提升，不能只靠单条轨迹拉高平均值。
3. 至少 7/8 probe 的 action L1 下降；同时检查逐 slot，尤其 right gripper slot 13。
4. `memory_valid_ratio` 与每条 probe 的历史 mask 符合预期。
5. 最终至少抽查 44 个一轨迹一 probe 计划，不把训练 probe 当作 held-out。

这些门槛证明“当前 44 条采集轨迹可被联合记忆”，仍不证明新场景泛化、闭环安全或任务成功率。

## 10. 真实 smoke 结果

2026-07-19 在单张 H200 上完成 2-step 真实 smoke，launcher 状态为 `succeeded`，耗时
132.7 秒（包含模型加载和 step-0 的 8-probe diffusion suite）。训练进程实际报告：

```text
cases=127
windows=486
samples_per_epoch=486
train/val dataset size=486/486
camera rectification mask=[true,true,true]
memory_valid_ratio=1.0
```

Stage-2 初始化的 8-probe step-0 均值为：

| 指标 | 值 |
|---|---:|
| val loss | 0.87821 |
| rollout vs GT PSNR | 9.08738 dB |
| rollout vs GT SSIM | 0.15276 |
| action L1 | 0.35710 |
| action MSE | 0.25557 |

8 条 probe 全部生成三联诊断、imagination、execution、action 和组合视频。组合视频已经通过
容器与像素检查：H.264、640x640、5 FPS、21 帧；上方是两个 320x416 标注视频，下方是
640x224 动作图。smoke 最终写出 `step_000002.pt`，证明 rectified full-dataset loader、backward、
optimizer 和 artifact 编码链路均可执行。

正式 10,000-step run 输出目录为：

```text
work/tuning/runs/fastwam_tianji_dataset_overfit
```

smoke 结果只建立 step-0 基线和可执行性，不计作收敛证据。正式结果必须在同一 8-probe suite
上比较 step 2,000、4,000、6,000、8,000、10,000。
