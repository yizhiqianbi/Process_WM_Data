# FastWAM 统一数据对齐与训练调试手册

更新日期：2026-07-15

本文记录从 `TrainingCaseV1` 到 FastWAM 训练 batch 的完整契约、实际代码路径、
RoboCOIN 真实数据验证结果、checkpoint 恢复结果，以及其余异构数据集接入时必须满足的条件。
当前范围只包含 FastWAM，不涉及 Cosmos。

## 1. 当前结论

FastWAM 已经可以直接训练本目录产出的 canonical sidecar，不需要先伪装成 LeRobot：

- 真实输入：RoboCOIN 两个 episode，四路原始 AV1 相机，canonical Parquet。
- 当前训练选用三路语义相机：`global_primary`、`left_wrist`、`right_wrist`。
- 控制时间轴：20 Hz，81 个 state 点，80 个 action transition，共 4 秒。
- 视觉时间轴：canonical offset `0,4,...,80`，共 21 帧，满足 FastWAM 的 `T % 4 == 1`。
- action/state：固定 80 维，RoboCOIN 实际有效 20 维，其余 60 维严格屏蔽。
- 训练验证：真实 Wan2.2-TI2V-5B + ActionDiT 完成 forward、backward、AdamW step。
- 恢复验证：从 step 1 恢复 model、optimizer、scheduler、sampler、RNG 后完成 step 2。
- 全数据验证：当前两个 episode 展开为 27 个窗口，27/27 均完成视频解码和 contract 检查。

这证明的是训练链路已经打通，不代表 RoboCOIN 全量数据已经完成清洗。当前 manifest 只含用于
验证的两个 episode；扩大训练规模时应重新运行完整 pipeline 并重建统计量与 text cache。

## 2. 代码与产物位置

FastWAM 代码：

```text
/path/to/FastWAM
├── src/fastwam/datasets/training_case_dataset.py
├── src/fastwam/models/wan22/fastwam.py
├── src/fastwam/models/wan22/fastwam_idm.py
├── scripts/precompute_text_embeds.py
├── scripts/validate_training_case_data.py
├── configs/data/canonical_robocoin_81f.yaml
├── configs/model/fastwam_joint_canonical.yaml
├── configs/task/canonical_robocoin_joint_81f_smoke.yaml
└── configs/task/canonical_robocoin_joint_81f_1e-5.yaml
```

数据与验证产物：

```text
/path/to/Process_WM_Data/work
├── validation/training_case_v1/robocoin
│   ├── cases/training_cases.jsonl
│   └── canonical/episodes/*/canonical.parquet
└── fastwam_training/robocoin_81f
    ├── normalization_stats.json
    ├── text_embeds/*.pt
    ├── data_validation.json
    └── runs
        ├── smoke/checkpoints            # step 1 -> step 2 resume proof
        └── smoke_final/checkpoints      # final code/stats single-step proof
```

## 3. 从 TrainingCaseV1 到 batch

### 3.1 样本展开

manifest 的一行对应一个 episode case，而不是一个训练样本。loader 展开：

```text
sampling.valid_starts = {start, stop_exclusive, stride, count}
case + window_start -> one FastWAM sample
```

当前两个 case 分别产生 15 和 12 个窗口，共 27 个。loader 会核对声明的 `count` 和真实展开
数量；时间轴、维度、schema version 或窗口边界不一致时直接失败，不随机换样。

### 3.2 时间对齐

对窗口起点 `s`：

```text
proprio = canonical_state[s : s + 80]       # [80, 80]
action   = canonical_action[s : s + 80]     # [80, 80]
video canonical rows = s + [0,4,...,80]     # 21 rows
video source frames = source_nearest_frame_index[those rows]
```

`action[t]` 的定义必须是 `state[t] -> state[t+1]`。因此窗口使用 81 个 state 点，但只向模型
提供与 80 个 transition 对齐的前 80 个 proprio 点。统计量计算同样排除每个 episode 最后一行
action，因为最后一行没有后继 state。

### 3.3 视频解码与布局

loader 使用 PyAV 按 `source_nearest_frame_index` 精确解码，不使用视频时间戳的近似 seek。
支持普通文件和 `tar://ARCHIVE!MEMBER`。三路图像按官方 RobotWin 布局组成：

```text
global_primary: resize 256 x 320
left_wrist:     resize 128 x 160
right_wrist:    resize 128 x 160

final frame:
+----------------------------------+
|          global 256x320          |
+----------------+-----------------+
| left 128x160   | right 128x160   |
+----------------+-----------------+
              384 x 320
```

最后输出 `[C,T,H,W] = [3,21,384,320]`，像素从 `[0,1]` 变换到 `[-1,1]`。
若异构数据缺少 wrist，相应区域默认填零，并通过 `camera_present_mask` 记录；也可配置
`missing_camera_policy=repeat_global`。全局相机缺失则拒绝样本。

### 3.4 80 维 canonical action/state

canonical 值和 mask 同时读取。每个窗口的有效维是以下三者交集：

1. manifest 的 `state_slot_mask` / `action_slot_mask`；
2. 窗口内每个时刻的 `*_dim_valid_mask`；
3. normalization stats 中该 domain 的 `valid_mask`。

无效维在进入模型前强制为零。action diffusion noise 对无效维也强制为零，且这些维不进入
action MSE 分子或分母。这样 80 维空间可以容纳不同本体，而不会把“填充零”误当成监督。

### 3.5 normalization

`scripts/build_fastwam_normalization_stats.py` 只读取 train split，并按
`embodiment.normalization_domain` 和 active canonical slot 计算 z-score：

```text
x_normalized = clip((x - mean) / max(std, 1e-3), -10, 10)
```

统计按 episode 去重，不能按窗口重复累计，否则长 episode 或高重叠窗口会被重复加权。
validation/test 必须复用 train stats，禁止重新拟合。

### 3.6 文本条件

指令统一包装为：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

`precompute_text_embeds.py` 已支持直接扫描 `case_manifests` 中的 `language.primary`，输出
Wan2.2 T5 context `[128,4096]` 和 mask。训练时 `load_text_encoder=false`，避免每个 worker
重复加载 11 GB text encoder。

## 4. 传给 FastWAM 的完整字段

| 字段 | shape | 作用 |
|---|---:|---|
| `video` | `[3,21,384,320]` | 三相机 composite，范围 `[-1,1]` |
| `action` | `[80,80]` | normalized canonical action |
| `proprio` | `[80,80]` | normalized canonical state |
| `context` | `[128,4096]` | 预计算 T5 embedding |
| `context_mask` | `[128]` | 文本 mask |
| `image_is_pad` | `[21]` | 视觉时间 padding |
| `action_is_pad` | `[80]` | action 时间 padding |
| `action_dim_is_pad` | `[80]` | action canonical 维度 padding |
| `proprio_dim_is_pad` | `[80]` | state canonical 维度 padding |
| `proprio_condition_mask` | scalar | 是否允许 state condition |
| `action_loss_mask` | scalar | A tier 为真，B tier 为假 |
| `video_loss_mask` | scalar | 是否计算视频 loss |
| `action_loss_weight` | scalar | case action 权重 |
| `video_loss_weight` | scalar | case video 权重 |
| `sample_weight` | scalar | `sample_weight * mixture_weight` |
| `camera_present_mask` | `[3]` | 三个语义位置是否有真实相机 |

DataLoader collate 后，除 prompt 外所有字段增加 batch 维。

## 5. FastWAM loss 改动

原实现只支持 `action_is_pad [B,T]`，并先对 action 维度求平均。这会让异构本体的空维进入
监督。当前实现的 action 有效张量为：

```text
valid[B,T,D] = ~action_is_pad[B,T,1] & ~action_dim_is_pad[B,1,D]
```

每个样本只在有效 `(T,D)` 上平均，再应用 diffusion scheduler weight、case loss weight、
sample/mixture weight 和样本级 loss mask。video loss 采用同样的样本级加权规约。

`FastWAM`、`FastWAMJoint`（继承基础实现）和 `FastWAMIDM` 都使用该逻辑。旧数据集不提供
新增字段时，默认全部有效且权重为 1，因此保持原行为。

## 6. 从零复现

### 6.1 环境

```bash
FASTWAM_REPO=${FASTWAM_REPO:-/path/to/FastWAM}
cd "$FASTWAM_REPO"
PY=${PYTHON_BIN:-python3}
$PY -m pip install -e . --no-deps

export FASTWAM_PREPROCESS_ROOT=/path/to/Process_WM_Data
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/model/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_DIT_PATH="$DIFFSYNTH_MODEL_BASE_PATH/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
```

`DIFFSYNTH_MODEL_BASE_PATH` 需要包含 Wan2.2-TI2V-5B shards、converted VAE、T5 和 tokenizer。
若权重迁移到新目录，只需改这两个环境变量，不需要修改 YAML。

### 6.2 重新计算 train stats

```bash
cd $FASTWAM_PREPROCESS_ROOT
$PY scripts/build_fastwam_normalization_stats.py \
  --manifest work/validation/training_case_v1/robocoin/cases/training_cases.jsonl \
  --data-root $FASTWAM_PREPROCESS_ROOT \
  --output work/fastwam_training/robocoin_81f/normalization_stats.json
```

### 6.3 生成 text cache

选择空闲 GPU，例如物理 GPU 5：

```bash
cd "$FASTWAM_REPO"
CUDA_VISIBLE_DEVICES=5 $PY scripts/precompute_text_embeds.py \
  task=canonical_robocoin_joint_81f_smoke +overwrite=false
```

### 6.4 验证全部窗口

```bash
$PY scripts/validate_training_case_data.py \
  task=canonical_robocoin_joint_81f_1e-5 \
  +validate_all=true \
  +validation_report=$FASTWAM_PREPROCESS_ROOT/work/fastwam_training/robocoin_81f/data_validation.json
```

预期关键结果：`windows=27`、`validated_windows=27`、video `[3,21,384,320]`、
action/proprio `[80,80]`、active dimensions `20/20`、camera mask `[true,true,true]`。

### 6.5 单步真实训练

```bash
CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py task=canonical_robocoin_joint_81f_smoke
```

2026-07-15 使用最终代码和重建 stats 的 `smoke_final` 实测：

```text
loss=0.8690  loss_action=0.8454  loss_video=0.0237
```

随机 diffusion timestep 会改变 loss 数值，因此验收条件是 loss 有限、backward/optimizer 成功、
权重与 state checkpoint 完整写出，不要求复现完全相同的数值。

### 6.6 checkpoint 恢复

```bash
CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py \
  task=canonical_robocoin_joint_81f_smoke \
  max_steps=2 \
  resume=$FASTWAM_PREPROCESS_ROOT/work/fastwam_training/robocoin_81f/runs/smoke/checkpoints/state/step_000001
```

实测恢复了 `epoch=0, batch_in_epoch=1, sample_offset=1`，随后完成 step 2。Accelerate 保存时会
提示 `model.dit` 与 `model.mot` 存在 shared tensors；这是同一 MoT 模块的兼容别名。实际从该
state 完整恢复已经通过，不能仅凭警告删除 checkpoint。

### 6.7 当前完整窗口训练配置

```bash
CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py task=canonical_robocoin_joint_81f_1e-5
```

该配置使用 27 个窗口、batch size 1、gradient accumulation 4、10 epochs、bf16 和 gradient
checkpointing。它是链路基线，不是最终多数据集超参数；加入更多 manifest 后应重新确定采样权重、
global batch size、学习率、warmup 和 checkpoint 周期。

## 7. 接入其余数据集

每个数据集必须先产生满足以下条件的 TrainingCaseV1：

1. canonical sidecar 确实有 81 个 state 点可组成窗口；
2. `source_nearest_frame_index` 能映射到可解码视频帧；
3. 五个 camera slot 使用语义角色，而不是目录排序；
4. action mapping 已验证才允许 A tier；未验证数据只能 B tier 且 action loss 为假；
5. normalization domain 对同一机器人、控制语义和 active slots 保持稳定；
6. split 按 lineage/group 划分，不能让同源 clip 跨 train/val；
7. train manifests 更新后重新计算 stats 和 text cache。

多数据集配置只需把 `case_manifests` 扩展为多个 JSONL，并把 `allowed_modes` 设置为需要的层级：

```yaml
case_manifests:
  - /path/to/robocoin/cases/training_cases.jsonl
  - /path/to/galaxea/cases/training_cases.jsonl
  - /path/to/oxe/cases/training_cases.jsonl
allowed_modes: [joint_video_action, video_only]
allowed_quality_tiers: [A, B]
```

B tier 仍会经过 video branch，但 `action_loss_mask=false`。若 `conditioning_mask.state=false`，
proprio token 会被置零并在 context mask 中关闭。

## 8. 当前限制与下一步

- 本文记录的 optimizer/checkpoint 基线只覆盖 RoboCOIN 两个验证 episode；AgiBot proprio 的
  真实 data/memory loader 回归见 `AGIBOT_PROPRIO_TRAINING.md`，尚待 GPU 单步 optimizer 验收。
- tar 内视频已可正确解码，但压缩 archive 的随机访问成本高；大规模训练应先做本地视频 shard/cache。
- 当前 production 配置默认关闭 rollout eval，以控制 21 帧扩散采样的成本。canonical evaluator
  已保留维度/时间 mask 并可报告 normalized-space action L1/L2；若需要物理单位指标，还需为每个
  normalization domain 实现 canonical slot 到原生控制量的可逆映射。
- 尚未加入图像增强、相机 dropout、重复窗口去相关和跨数据集 batch sampler。
- 80 维 action head 的 encoder/head 是随机初始化，ActionDiT 只加载 backbone；这是 checkpoint
  policy 的既定行为，正式训练需要足够 warmup 和数据量。
- 全量扩展前应先分别统计 A/B/C 数量、domain 数量、有效维分布、相机组合和窗口重叠率，避免
  大数据集或高帧率数据集在 mixture 中失控。

## 9. 验收清单

- [x] TrainingCaseV1 schema 和 81/80/21 timeline 严格检查。
- [x] canonical Parquet window、action/state mask 和 domain stats 对齐。
- [x] 三相机 AV1 精确解码与 `[3,21,384,320]` composite。
- [x] T5 cache 从 manifest 指令生成并加载。
- [x] action temporal/dimension/sample 三层 mask 进入 loss。
- [x] 真实 5B forward、backward、optimizer step。
- [x] 权重 checkpoint 和完整训练状态 checkpoint。
- [x] 从 step 1 恢复并继续到 step 2。
- [x] 当前 27/27 窗口全量解码验证。
- [ ] 七个数据集的全量 TrainingCaseV1 生成与分布审计。
- [x] canonical mask-aware validation/inference evaluator（normalized action 指标）。
- [ ] 按 normalization domain 反归一化到物理单位的 evaluator。
- [ ] 多数据集分层 sampler 与正式长跑配置。
