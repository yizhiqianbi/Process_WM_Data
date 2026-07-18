# FastWAM 三阶段预训练、Memory 训练与专项微调方案

更新日期：2026-07-18

本文是当前九个逻辑机器人数据集接入 FastWAM 的执行手册。范围固定为 FastWAM，不包含
Cosmos。文档区分“代码路径可运行”和“动作标签可用于学习”两个概念，不能因为视频能解码
就把未知语义的 action 填进动作损失。

## 1. 结论与分层架构

训练采用三个阶段：

1. **Stage 1: video backbone pretrain**。使用九个逻辑数据集的 A/B 级视频、语言和相机布局，
   只训练 Wan2.2-TI2V-5B video DiT，不训练 ActionDiT。目标是得到适应机器人视角、手眼
   运动、接触和场景变化的视频 backbone。
2. **Stage 2: memory FastWAM pretrain**。从 Stage 1 初始化 video expert，同时加载已有
   ActionDiT backbone，训练魔改后的 `MemoryFastWAM`。A 级样本同时计算 video/action loss；
   B 级样本只计算 video loss，但仍真实经过 memory、MoT 和 video expert 路径。
3. **Stage 3: dataset/task-specific finetune**。从 Stage 2 checkpoint 恢复完整 MoT、
   memory patcher 和 proprio encoder，在目标本体或目标任务上使用较小学习率微调。

当前最重要的原则是：

- 原七个数据集已进入 Stage 1、MemoryFastWAM，并完成逐库真实解码验收。
- 原七库 Stage 1/2 已使用 dataset-balanced sampler 重新完成真实 5B/6B 训练回归。
- LingBot-VA 和 DreamZero 已完成固定 SHA 真实数值样本的 81/80/21 数据链路；全量视频
  decode、FastWAM loader 和 GPU optimizer/checkpoint 回归仍待执行。
- MemoryFastWAM action inference 已使用真实 8/2/1 历史和 Stage 3 checkpoint 验收。
- 当前预处理实测中，OXE ASU、OXE-AugE、AgiBot-Beta、RoboCOIN、RoboMIND、Galaxea 和
  InternData-A1 都已产出动作监督 A 级 case；AgiBot 当前验证范围是 task 389 / episode 673828。
- B 级 case 的 action/state 张量为零且所有维度 mask 为无效，`action_loss_mask=false`。
- 任何数据集只有在本体映射、时间对齐和 action 语义确认后，才允许升级到 A 级。
- 上述九库结论是样本级数据预处理验收结果；当前重建的 validation normalization stats
  覆盖 RoboCOIN 和 AgiBot 两个 domain，全量九库 stats 和联合 optimizer/checkpoint 回归
  尚未重跑。

## 2. 统一训练样本

每个训练样本使用固定时间合同：

```text
target rate:       20 Hz
state points:      81
action transitions:80
duration:          4 seconds
video offsets:     0,4,8,...,80
video frames:      21
video tensor:      [3,21,384,320]
state tensor:      [80,80]
action tensor:     [80,80]
text context:      [128,4096]
```

第 81 个 state 只用于定义第 80 个 transition 的终点。模型输入的 proprio 是前 80 个
state，action 定义为 `state[t] -> state[t+1]`。所有源数据先映射到 20 Hz canonical
timeline，再根据 `source_nearest_frame_index` 读取图像。

相机使用语义角色，不使用文件夹顺序：

```text
+----------------------------------+
| global_primary       256 x 320   |
+----------------+-----------------+
| left_wrist     | right_wrist     |
| 128 x 160      | 128 x 160       |
+----------------+-----------------+
final composite: 384 x 320
```

缺 wrist 时默认填零，并通过 `camera_present_mask` 标记。缺全局主相机时 case 不应进入训练。

### 2.1 解码合并与 tar 视频缓存

`MemoryTrainingCaseDataset` 不再分别解码当前 21 帧和 memory 11 帧。同一个样本会先合并
两组 source frame index，每个相机只打开一次底层视频，然后按顺序拆回：

```text
current video: 21 frames
memory video:  8 + 2 + 1 frames
decoder open:  1 time per unique camera URI
```

对 AgiBot-Beta、Galaxea 的 `tar://ARCHIVE!MEMBER`，直接从 `.tar.gz` 读取会在每个 sample
重复扫描和解压 archive。Stage 1/2 配置已启用：

```text
work/stage_pipeline/video_member_cache/
```

cache key 包含 archive 绝对路径、archive size、mtime 和 member name；成员写入临时文件并
通过 `os.replace` 原子提交，多 worker 同时命中不会留下半文件。archive 改变后 key 自动
变化。cache 保存的是原始 MP4 member，不做重编码，不修改下载数据。

正式训练前可预热当前 TrainingCase manifest 引用的 tar 视频：

```bash
REPO=${FASTWAM_REPO:-/path/to/FastWAM}
PRE=${FASTWAM_PREPROCESS_ROOT:-$(pwd)}
PY=${PYTHON_BIN:-python3}

cd "$REPO"
$PY scripts/prewarm_tar_video_cache.py \
  --case-manifest "$PRE/work/stage_pipeline/agibot_beta/cases/training_cases.jsonl" \
  --case-manifest "$PRE/work/stage_pipeline/galaxea/cases/training_cases.jsonl" \
  --data-root "$PRE" \
  --cache-dir "$PRE/work/stage_pipeline/video_member_cache" \
  --workers 3
```

当前 9 个真实 tar video member 首次物化耗时 `129.50s`；再次运行全部 cache hit，耗时
`4.66s`。启用 cache 后，七库 7 个 memory case 的完整 CPU 解码和张量验收耗时 `7.46s`。
全量预热会额外占用接近所引用压缩视频 member 总大小的空间，启动前应统计 cache 预算。

## 3. 九个逻辑数据集当前验收矩阵

下表对应 `work/stage_pipeline/*` 中当前实际样本，不代表全量下载规模。

| 数据集 | 当前原生读取路径 | 窗口数 | 当前三相机 mask | 数据级别 | Stage 1 | Memory 路径 | action 学习 |
|---|---|---:|---|---|---|---|---|
| OXE | tar 内受限 NumPy pickle | 34 | `1,0,0` | A/B window | 通过 | 通过 | ASU 7 维有效 |
| OXE-AugE | target replay Parquet/视频 | 11 | `1,0,0` | A | 通过 | 通过 | 7/8 维派生 target |
| AgiBot-Beta | observation tar + proprio HDF5 | 18 | `1,1,1` | A | 通过 | 通过 | 20 维有效 |
| RoboCOIN | LeRobot Parquet + AV1 MP4 | 15 | `1,1,1` | A | 通过 | 通过 | 20 维有效 |
| RoboMIND | HDF5 内 JPEG 图像 | 2 | `1,0,0` | A | 通过 | 通过 | 当前 UR 7 维有效 |
| Galaxea | tar 内 LeRobot Parquet/AV1 | 24 | `1,1,1` | A | 通过 | 通过 | 当前 26 维有效 |
| InternData-A1 | LeRobot Parquet/视频 | 8 | `1,1,1` | A | 通过 | 通过 | 20 维有效 |
| LingBot-VA | RoboTwin/LIBERO LeRobot | 3/1 | `1,1,1` / `1,1,0` | A | 待全视频验收 | 待回归 | 14/7 维有效 |
| DreamZero-DROID | LeRobot + GEAR modality | 6 | global x2 + wrist | A | 待全视频验收 | 待回归 | 8 维有效 |

原七库逐库验收选择第一个 `window_start > 0` 的窗口，且均为 `window_start=40`，因此
memory 取 canonical index 29 至 39，11 个 index 全部小于 40。所有样本均得到有限值的
`[3,21,384,320]` 当前视频和 8/2/1 memory 视频。新两库当前只完成预处理 TrainingCase
验收，不包含在这项 FastWAM tensor 结论内。

机器可读报告生成命令：

```bash
PRE=${FASTWAM_PREPROCESS_ROOT:-$(pwd)}
PY=${PYTHON_BIN:-python3}
cd "$PRE"
$PY scripts/validate_fastwam_training_cases.py
```

报告位置：

```text
work/stage_pipeline/validation/all_datasets.json
```

## 4. Stage 1: Video Backbone

### 4.1 输入和损失

Stage 1 使用：

- `video [B,3,21,384,320]`
- `context [B,128,4096]`
- `context_mask [B,128]`
- `image_is_pad [B,21]`
- `video_loss_mask`、`video_loss_weight`、`sample_weight`

action/proprio 仍由 loader 生成固定 shape，但全部 mask 为无效，不进入 video backbone。
`load_text_encoder=false`，训练 worker 只读取预计算 UMT5 context，避免每个进程加载约
11 GB 文本编码器。

### 4.2 参数更新范围

`train_scope=dit`：

- 训练 Wan video DiT；
- 冻结 VAE；
- 不创建 ActionDiT；
- 不加载文本编码器；
- video attention 使用 `first_frame_causal`；
- gradient checkpointing 开启。

Stage 1 checkpoint 至少包含：

```text
dit: video DiT state_dict
step: global step
torch_dtype: model dtype
```

Stage 2 通过 `FASTWAM_STAGE1_VIDEO_CKPT` 严格加载 `dit`，缺 key 或 shape 不匹配应直接失败。

### 4.3 配置和命令

单步 Smoke 配置：`configs/task/stage1_video_backbone_smoke.yaml`

七库各一步配置：`configs/task/stage1_all_datasets_smoke.yaml`

生产配置：`configs/task/stage1_video_backbone_pretrain.yaml`

生产配置默认 `sampling_strategy=dataset_balanced`。每个 epoch 的总 sample 数仍等于 combined
dataset 长度，但七库配额相差不超过 1；小库按随机无放回循环补齐，大库跨 epoch 随机轮换。
sampler 使用 `seed + epoch`，并保留全局 batch offset 恢复语义。`sample_weight` 仍只控制
loss 权重，不再被误当作采样概率。

```bash
REPO=${FASTWAM_REPO:-/path/to/FastWAM}
PRE=${FASTWAM_PREPROCESS_ROOT:-$(pwd)}
PY=${PYTHON_BIN:-python3}

export FASTWAM_PREPROCESS_ROOT="$PRE"
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/model/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true

cd "$REPO"
CUDA_VISIBLE_DEVICES=5 $PY scripts/precompute_text_embeds.py \
  task=stage1_video_backbone_smoke +overwrite=false

CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py task=stage1_video_backbone_smoke

# 每库恰好一个样本，共 7 个 optimizer step
CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py task=stage1_all_datasets_smoke
```

当前 smoke 实测完成 1 次 optimizer step：

```text
step=1/1
loss_video=0.0206
checkpoint=work/stage_pipeline/runs/stage1_video_smoke/checkpoints/weights/step_000001.pt
```

七库逐库 smoke 使用 `max_samples_per_case=1`，数据集长度严格为 7，完成 7/7 optimizer
step。七步 `loss_video` 分别为 `0.0175, 0.0792, 0.3514, 0.2644, 0.1452, 0.1122,
0.3076`，最终 checkpoint 为：

```text
work/stage_pipeline/runs/stage1_all_datasets_smoke/checkpoints/weights/step_000007.pt
```

生产训练的 100k step 只是初始预算，不应把它理解为固定最优值。正式启动前仍需要按全量
manifest 重算每库窗口数、审计 balanced 后的重复率，并确定验证集和算力预算。

## 5. Stage 2: MemoryFastWAM

### 5.1 初始化关系

```text
Stage 1 video checkpoint -> video expert
pretrained ActionDiT      -> action expert backbone
random action encoder/head-> canonical 80D action interface
video patch embedding     -> initialize three memory patchers
```

memory patcher 使用 video patch embedding 初始化：

| memory group | 输入帧数 | Conv3d kernel/stride | 作用 |
|---|---:|---|---|
| long | 8 | `(4,8,8)` | 最强时空压缩 |
| mid | 2 | `(2,4,4)` | 中等压缩 |
| short | 1 | `(1,2,2)` | 保留最近帧细节 |

这里的 long/mid/short 是 token 化分辨率分层。历史帧本身按时间连续排列：取当前窗口之前
最近 11 个 canonical observation，先 8 个进入 long，再 2 个进入 mid，最近 1 个进入
short。这与恢复的 Helios memory 8/2/1 合同一致。

### 5.2 因果性与早期窗口

当前窗口起点为 `s` 时，合法 memory index 必须满足：

```text
0 <= memory_index < s
```

绝不允许读取当前 81-step target window 内的任何图像。episode 开头不足 11 帧时，左侧
使用当前 observation 作为 shape padding，但对应 `memory_mask=false`。VAE 后再次乘 mask，
memory token 也乘 mask；MoT 使用逐样本 `[B,1,S,S]` attention mask，防止 Conv3d bias
产生的非零 padding token 被 target/action 消费。

### 5.3 Attention 可见性

- 有效 memory query 只看有效 memory key。
- target video query 可看有效 memory，并按现有 video causal mask 看 target video。
- action query 可看有效 memory、当前视频第一帧和 action token。
- 无效 memory key 对 target/action 均不可见。
- 旧 FastWAM 的共享 `[S,S]` mask 继续支持；memory 使用 `[B,1,S,S]`。

### 5.4 Memory-aware action inference

`MemoryFastWAM.infer_action` 已独立实现，不再继承基础 FastWAM 后静默丢弃 memory。推理时：

- 推荐显式传入 `memory_video_long/mid/short` 和三个 mask；
- 也可传入 closed-loop `memory_video [1,3,T,H,W]`，最后一帧必须是当前 observation，函数
  会排除当前帧并生成 8/2/1；
- episode 冷启动未提供 history 时，会构造全 invalid memory，但仍保留与训练一致的 token
  layout 和 target RoPE 起点；
- video KV cache 支持逐样本 `[B,1,S,S]` mask，无效 padding key 不可被 action 消费；
- memory joint video/action inference 尚未实现，`infer_joint` 会明确抛错，不会静默走基础路径。

真实 checkpoint 验收命令：

```bash
STAGE3_CKPT="$PRE/work/stage_pipeline/runs/stage3_from_balanced_io_regression_20260716/checkpoints/weights/step_000001.pt"
CUDA_VISIBLE_DEVICES=5 $PY scripts/validate_memory_inference.py \
  task=stage3_robocoin_memory_smoke \
  +validation_checkpoint="$STAGE3_CKPT" \
  +validation_num_inference_steps=1
```

当前使用 Stage 3 checkpoint 和 RoboCOIN `window_start=40` 的结果为：

```text
status=passed
action_shape=[80,80]
memory_valid_ratio=1.0
action_range=[-4.09375,4.25]
```

### 5.5 A/B 样本如何混训

统一 Stage 2 dataset 接受 A/B 两级：

```text
A: loss = lambda_video * video_loss + lambda_action * action_loss
B: loss = lambda_video * video_loss
```

B 级样本即使 canonical sidecar 中存在原始 action 列，也不会加载这些值：loader 根据
`training.mode=video_only` 返回全零 action/state 和全 invalid dimension mask。这是防止
未知本体语义污染 ActionDiT 的硬约束。

当前全库配置是 `configs/data/canonical_stage2_memory_all.yaml`。stats 构建器会从 pipeline
自动发现 manifest，并只统计 `train + A + joint_video_action`。当前小样本文件只包含
RoboCOIN 和 AgiBot 两个 domain；其他已开放 domain 必须在全量 train split 处理完成后重新
生成，不能借用相邻本体或当前 validation stats 进入正式 Stage 2。B 级样本不查询
action/state stats。

### 5.6 参数组

`train_scope=model_configured`：

- video/action MoT experts 和 proprio encoder 使用 `learning_rate`；
- 43.066M memory patcher 参数使用 `reference_learning_rate`；
- VAE 和文本编码器冻结；
- 当前 MoT trainable 参数约 6021.155M。

### 5.7 配置和命令

RoboCOIN 动作监督 smoke：`configs/task/stage2_memory_fastwam_smoke.yaml`

非空历史 smoke：`configs/task/stage2_memory_active_smoke.yaml`

七库各一步 smoke：`configs/task/stage2_all_datasets_memory_smoke.yaml`

七库生产配置：`configs/task/stage2_memory_fastwam_pretrain.yaml`

```bash
export FASTWAM_STAGE1_VIDEO_CKPT="$PRE/work/stage_pipeline/runs/stage1_video_smoke/checkpoints/weights/step_000001.pt"
CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py task=stage2_memory_fastwam_smoke

# 使用 Stage 1 七库 checkpoint，七库各取一个非空 memory 窗口
export FASTWAM_STAGE1_VIDEO_CKPT="$PRE/work/stage_pipeline/runs/stage1_all_datasets_smoke/checkpoints/weights/step_000007.pt"
CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py task=stage2_all_datasets_memory_smoke
```

当前实测结果：

```text
stage2_memory_smoke:
  loss_action=0.9946, loss_video=0.0161
  memory_tokens=168, memory_valid_ratio=0.0   # episode 起点 padding case

stage2_memory_active_smoke:
  loss_action=0.7909, loss_video=0.0191
  memory_tokens=168, memory_valid_ratio=1.0   # window_start=40，11/11 history 有效
  checkpoint=work/stage_pipeline/runs/stage2_memory_active_smoke/checkpoints/weights/step_000001.pt
```

历史七库 memory smoke 使用 `sample_offset_per_case=1` 和 `max_samples_per_case=1`，严格得到
七个 `window_start=40` 的不同数据源样本。该运行发生在 AgiBot proprio 接入之前，当时六个
B 级 step 的 `loss_action=0`，唯一 RoboCOIN A 级 step 的 `loss_action=2.9899`。它仍证明了
memory 和 checkpoint 路径，但不能作为当前七库 action 配置的最终回归。最终 checkpoint：

```text
work/stage_pipeline/runs/stage2_all_datasets_memory_smoke/checkpoints/weights/step_000007.pt
```

`FASTWAM_STAGE1_VIDEO_CKPT` 现在是强制环境变量；缺失时 Hydra 配置解析直接失败，避免
Stage 2 意外从基础 Wan 权重开始。生产运行时将它改为 Stage 1 正式 checkpoint，再执行：

```bash
CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py task=stage2_memory_fastwam_pretrain
```

## 6. Stage 3: 指定数据集微调

Stage 3 不重新初始化模型。必须同时提供：

- Stage 1 checkpoint，用于构造 video expert 的基础状态；
- Stage 2 checkpoint，用于覆盖完整 MoT、memory patchers 和 proprio encoder。

生产和 smoke 配置都强制解析 `FASTWAM_STAGE1_VIDEO_CKPT` 与
`FASTWAM_STAGE2_MEMORY_CKPT`。Memory checkpoint 使用 strict MoT load，并检查 checkpoint
中的 history size 是否仍为 8/2/1；任何缺 key、额外 key、shape 或 history 合同不一致都会
在训练前失败。

RoboCOIN 生产配置：`configs/task/stage3_robocoin_memory_finetune.yaml`

RoboCOIN smoke 配置：`configs/task/stage3_robocoin_memory_smoke.yaml`

```bash
export FASTWAM_STAGE1_VIDEO_CKPT="$PRE/work/stage_pipeline/runs/stage1_video_smoke/checkpoints/weights/step_000001.pt"
export FASTWAM_STAGE2_MEMORY_CKPT="$PRE/work/stage_pipeline/runs/stage2_memory_smoke/checkpoints/weights/step_000001.pt"
CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py task=stage3_robocoin_memory_smoke
```

后续推荐使用七库验收 checkpoint 组成同一条链：

```bash
export FASTWAM_STAGE1_VIDEO_CKPT="$PRE/work/stage_pipeline/runs/stage1_all_datasets_smoke/checkpoints/weights/step_000007.pt"
export FASTWAM_STAGE2_MEMORY_CKPT="$PRE/work/stage_pipeline/runs/stage2_all_datasets_memory_smoke/checkpoints/weights/step_000007.pt"
CUDA_VISIBLE_DEVICES=5 $PY scripts/train.py task=stage3_robocoin_memory_smoke \
  output_dir="$PRE/work/stage_pipeline/runs/stage3_robocoin_from_all_datasets"
```

当前 Stage 3 已从 Stage 2 `.pt` 成功恢复并完成一次 optimizer step：

```text
loss_action=0.8583
loss_video=0.0095
checkpoint=work/stage_pipeline/runs/stage3_robocoin_smoke/checkpoints/weights/step_000001.pt
```

最终又使用七库 Stage 1/2 checkpoint 组成同一条链，并固定选择 `window_start=40`：

```text
loss_action=0.6558
loss_video=0.0087
memory_tokens=168
memory_valid_ratio=1.0
checkpoint=work/stage_pipeline/runs/stage3_robocoin_from_all_datasets_active/checkpoints/weights/step_000001.pt
```

2026-07-16 在合并解码、tar cache 和 balanced sampler 修改后重新跑通同一条链：

```text
Stage 1 (七库 7/7):
  checkpoint=work/stage_pipeline/runs/stage1_balanced_io_regression_20260716/
             checkpoints/weights/step_000007.pt
  size=9,999,877,109 bytes

Stage 2 (七库 7/7):
  checkpoint=work/stage_pipeline/runs/stage2_balanced_io_regression_20260716/
             checkpoints/weights/step_000007.pt
  memory_tokens=168, memory_valid_ratio=1.0 for all seven samples
  six B-tier action losses=0; RoboCOIN action loss=0.8063
  size=12,129,102,406 bytes

Stage 3 (RoboCOIN):
  checkpoint=work/stage_pipeline/runs/stage3_from_balanced_io_regression_20260716/
             checkpoints/weights/step_000001.pt
  loss_action=0.5760, loss_video=0.0086, memory_valid_ratio=1.0
  size=12,129,102,406 bytes
```

Stage 3 默认学习率比 Stage 2 小：backbone `2e-6`，memory/reference `2e-5`。正式训练应增加
目标数据集 validation split，并至少监控：

- normalized action MSE/L1；
- 反归一化后的 joint/EE/gripper 单位误差；
- rollout success rate；
- video denoising loss；
- memory ablation（8/2/1 对比无 memory）；
- 不同 `window_start` 的性能，尤其 episode 早期 padding 场景。

## 7. Checkpoint 衔接与恢复

三个阶段使用不同的轻量权重合同：

| 阶段 | 轻量 `.pt` 主要字段 | 下阶段用途 |
|---|---|---|
| Stage 1 | `dit` | 初始化 video expert |
| Stage 2/3 | `mot`, `memory_patchers`, `proprio_encoder` | 恢复 MemoryFastWAM |

每次保存还会产生 Accelerate state：model safetensors、optimizer、scheduler、sampler、RNG 和
`trainer_state.json`。同一阶段无缝续训优先恢复 state 目录；跨阶段初始化使用轻量 `.pt`。

不要把 Stage 1 的 `.pt` 直接作为 Stage 2 `resume`：它只含 video `dit`。Stage 2 应通过
`FASTWAM_STAGE1_VIDEO_CKPT` 初始化；只有 MemoryFastWAM `.pt` 才能作为 Stage 3 `resume`。
跨阶段 `.pt` 恢复会从新阶段的 step 0 开始；同阶段无缝续训才使用 Accelerate state 目录。

## 8. 每个数据集升级到 Action A 级的条件

### OXE

- ASU UR5 已按官方 xyz+rpy schema 和本地 next-state 数值关系进入 A。
- 其余 OXE 子数据集仍需分别确认 action/state key，不能复用 ASU mapping。
- 每个子集明确 absolute/delta/velocity、坐标 frame 和控制频率后才能开放。
- 每个 embodiment 独立 normalization domain。

### OXE-AugE

- 已按 target robot 拆 episode，并把 `state[t+1]` 记录为有 provenance 的派生 target。
- 保留增强样本与源 episode 的 lineage，防止 train/val 泄漏。
- 训练统计和论文报告必须将 derived target 与 native command 分开。

### AgiBot-Beta

- 真实 task 389 / episode 673828 已完成 observation/proprio/task join，三路相机、20 个
  canonical state/action slot 和 18 个 `81/80/21` 窗口均进入 A。
- action group 的 `index` 数据集取交集；当前原始 1226 行筛为 index `24..1190` 的 1167 行，
  不能按数组前缀截断。
- gripper feedback 为毫米、command 为归一化闭合量，保留独立归一化但不参与同量纲 alignment；
  其余 18 个槽位 alignment score 1.0、lag 0。
- `parameters` 仅影响相机 calibration metadata，不是 action 训练依赖。剩余 proprio 分片和
  task_info 仍需续传并做全量统计。

单库 memory smoke 配置为 `configs/task/stage2_agibot_memory_smoke.yaml`。数据和 memory
loader 已验证；当前节点 CUDA/NVML 不可见且 8 张 H200 正被其他训练占用，因此本次没有
启动新的 5B optimizer step。GPU 可用后执行：

```bash
export FASTWAM_PREPROCESS_ROOT=/path/to/Process_WM_Data
export FASTWAM_STAGE1_VIDEO_CKPT=/path/to/stage1/checkpoints/weights/step_N.pt
export FASTWAM_ACTION_DIT_PATH=/path/to/ActionDiT/checkpoint
CUDA_VISIBLE_DEVICES=0 $PY scripts/train.py task=stage2_agibot_memory_smoke
```

### RoboCOIN

- 当前 20 个 canonical action/state slot 已验证并可训练。
- 全量扩展后重新做 train-only stats，不能复用当前小样本统计量。
- 对不同 robot/config 建独立 normalization domain。

### RoboMIND

- 官方配置表内五类本体已建立不同的 puppet/master action root；当前 UR 实样进入 A。
- 未命中官方表的本体继续 B 级，不从相邻本体外推。
- 全量检查遥操作 lag、BGR/RGB 约定和各摄像头键。

### Galaxea

- 当前 R1-lite 的 joint/gripper/chassis/torso mapping 和 alignment 已进入 A。
- 未进入 canonical 的 EE telemetry 只监控，不再错误阻断已验证 control signal。
- 全量仍需按本体标定单位、物理 limit 和 action/state lag。

### InternData-A1

- 当前真实 LeRobot trajectory、三路视频和 20D action/state mapping 已进入 A。
- 对 `real/sim/sim_updated` 分开建立 source profile 和 normalization domain。
- 全量统计各 A1 本体单位、控制延迟和 slot 覆盖率。

## 9. 正式训练前的硬门槛

1. 全量下载清单有 checksum/size，archive 不缺 part。
2. 全量 scan/clean/materialize/windows/cases 完成，C 级数据不进入训练。
3. 七库逐库运行 validation 脚本，不能只验证 combined dataset 的前 N 个窗口。
4. 所有 A 级 domain 的 normalization stats 只由 train split 计算。
5. text cache 覆盖全部去重 prompt。
6. Stage 1、2、3 各完成至少一次真实 forward/backward/optimizer/checkpoint/resume。
7. 多 GPU 正式运行前，用 2 GPU 做 sampler、梯度累积和 checkpoint 恢复验收。
8. 记录每库有效窗口数、balanced 后的重复率和 loss 权重；sampler 已均衡库级配额，但不能
   替代全量数据分布审计。

## 10. 代码和配置索引

数据侧：

```text
Preprocess_FastWAM/fastwam_preprocess/
Preprocess_FastWAM/scripts/run_pipeline.py
Preprocess_FastWAM/scripts/validate_fastwam_training_cases.py
Preprocess_FastWAM/work/stage_pipeline/<dataset>/
```

训练侧：

```text
FastWAM/src/fastwam/datasets/training_case_dataset.py
FastWAM/src/fastwam/datasets/memory_bank.py
FastWAM/src/fastwam/models/wan22/memory_fastwam.py
FastWAM/src/fastwam/models/wan22/mot.py
FastWAM/src/fastwam/utils/samplers.py
FastWAM/scripts/prewarm_tar_video_cache.py
FastWAM/scripts/validate_memory_inference.py
FastWAM/configs/data/canonical_stage1_all.yaml
FastWAM/configs/data/canonical_stage2_memory_all.yaml
FastWAM/configs/data/canonical_agibot_memory_81f.yaml
FastWAM/configs/data/canonical_robocoin_memory_81f.yaml
FastWAM/configs/task/stage1_video_backbone_pretrain.yaml
FastWAM/configs/task/stage2_memory_fastwam_pretrain.yaml
FastWAM/configs/task/stage2_agibot_memory_smoke.yaml
FastWAM/configs/task/stage3_robocoin_memory_finetune.yaml
```

本方案此前已完成七库数据和 memory 输入验收、Stage 1 七库 5B 训练、历史配置下的 Stage 2
8/2/1 memory 训练、Stage 2 到 Stage 3 strict checkpoint 恢复、专项微调和真实
memory-aware action inference。2026-07-18 AgiBot proprio 接入后，七个数据集的当前实样都能生成
统一 81/80/21 action case，但尚未使用全量七域 stats 重跑 Stage 2。当前 manifest 仍是逐库真实样本级验收集，
不等同于全量 manifest 已生成或 100k step 正式预训练已经完成。动作证据和边界见
`ACTION_DATA_ADMISSION.md`。
