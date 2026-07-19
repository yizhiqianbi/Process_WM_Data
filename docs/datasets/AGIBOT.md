# AgiBot-Beta Proprio 清洗与 FastWAM 训练接入

[文档索引](../README.md)

更新日期：2026-07-18

本文给出 AgiBot-Beta 从官方 `proprio_stats` 到 MemoryFastWAM Stage 2 的完整可执行合同。
当前结论来自真实 task 389 / episode 673828，不是合成数组结果。全量七个 proprio 分片仍在
下载，因此本文区分“单 episode 链路验收通过”和“全量训练数据已经冻结”两种状态。

## 1. 当前验收结论

| 项目 | 真实结果 |
|---|---|
| 源 episode | `agibot_beta/local/agibot_g1/389/673828` |
| 原始 HDF5 行数 | 1226 |
| action index 交集 | 1167 行，原始 index `24..1190` |
| 估计源控制频率 | 约 29.986 Hz |
| canonical 时间线 | 786 state，785 action，20 Hz |
| FastWAM 窗口 | 18 个，全部 `joint_video_action` |
| 窗口合同 | 81 state / 80 action / 21 video |
| 相机 | head + left wrist + right wrist |
| 有效 state/action 维度 | 20 / 20 |
| quality | A |
| action loss | enabled |
| 8/2/1 memory | 11/11 有效且严格因果 |

FastWAM loader 的真实输出为：

```text
video                  [3, 21, 384, 320]
action                 [80, 80]
proprio                [80, 80]
context                [128, 4096]
memory_video_long      [3, 8, 384, 320]
memory_video_mid       [3, 2, 384, 320]
memory_video_short     [3, 1, 384, 320]
camera_present_mask    [true, true, true]
```

稳定的仓库级通过/待验矩阵见
[Validation Status](../reference/VALIDATION_STATUS.md)。运行中的下载字节数属于机器状态，不写入
Git；下载完成以 size-verified receipt 的 `status=ok` 为准。

## 2. 下载边界

已有 observation 视频时，Stage 2 必需的新增组件只有：

```text
AgiBot_Beta/AgiBotWorld-Beta/
├── observations/<task>/<range>.tar
├── proprio_stats/<range>.tar
└── task_info/task_<task>.json
```

`parameters/` 包含相机内外参，适合几何增强、跨相机投影和标定审计，但不是 action 训练的
硬依赖。不要为了等待约 1.5 TiB calibration 文件而阻塞 proprio 训练。

下载或续传全部 action-training 组件：

```bash
export FASTWAM_DATA_ROOT=/path/to/robot_dataset
export HF_TOKEN_FILE=/secure/path/hf_token.txt

python3 scripts/download_agibot_training_assets.py \
  --data-root "$FASTWAM_DATA_ROOT" \
  --token-file "$HF_TOKEN_FILE" \
  --file-workers 4
```

该命令固定使用 `configs/download_manifest.lock.json` 中的 revision，只选择七个
`proprio_stats` tar 和全部 `task_info`，完成后逐文件核对远端 size 并写原子 receipt。

## 3. Observation 与 proprio join

adapter 先建立两个 episode ID 索引：

```text
observations/<task>/<range>.tar
  └── <episode>/videos/*.mp4

proprio_stats/<range>.tar
  └── <task>/<episode>/proprio_stats.h5
```

join key 是字符串化 episode ID，不依赖 tar range 名。任务文本来自
`task_info/task_<task>.json`；当前 primary language 为 `Pickup in the supermarket`，子任务文本
保留在 alternatives 中。缺 task 文本会阻断 language-conditioned case，但不改变控制数组。

proprio tar 是未压缩 tar。扫描时记录每个 HDF5 member 的 `offset_data` 和 `size`；后续只 seek
并读取目标 HDF5 bytes，不会为每个 episode 重新遍历约 45 GiB 的 tar。测试要求同一 proprio
tar 在 adapter 扫描中只打开一次。

## 4. HDF5 schema 与 valid index

当前官方实样包含：

| 数据集 | shape | 语义 |
|---|---:|---|
| `timestamp` | `[1226]` | 纳秒时间戳 |
| `state/joint/position` | `[1226,14]` | 双臂关节反馈 |
| `action/joint/position` | `[1226,14]` | 双臂 HAL target |
| `state/effector/position` | `[1226,2]` | 双 gripper feedback |
| `action/effector/position` | `[1226,2]` | 双 gripper command |
| `state/head/position` | `[1226,2]` | head feedback |
| `action/head/position` | `[1226,2]` | head target |
| `state/waist/position` | `[1226,2]` | waist feedback |
| `action/waist/position` | `[1226,2]` | waist target |

不能用数组长度直接认定每行 action 有效。各 action group 可以提供独立的 `<group>/index`。
reader 执行以下规则：

1. 收集当前 HDF5 中所有已映射 action group 的 index 集合。
2. 对集合取 intersection。
3. 丢弃 `<0` 或超出所有读取数组 row count 的 index。
4. 使用原始 index 同步选取 timestamp、state 和 action。
5. `frame_index` 保留原始 index；timestamp 以首个有效时刻归零。
6. 交集为空或配置的 index key 缺失时 hard fail，不回退到前缀截断。

episode 673828 的 joint index 为 1167 个值 `24..1190`，其他已映射 group 覆盖全长，所以最终
交集同样为 `24..1190`。如果错误地读取前 1167 行，state/action 会相对视频提前 24 个源帧。

## 5. Canonical mapping

80D canonical 中当前激活：

| 原始字段 | source index | canonical slot | 数量 |
|---|---|---|---:|
| effector left/right | `0,1` | `6,13` | 2 |
| left arm joints | `0..6` | `14..20` | 7 |
| right arm joints | `7..13` | `21..27` | 7 |
| waist | `0,1` | `58,59` | 2 |
| head | `0,1` | `60,61` | 2 |

完整有效槽位为：

```text
[6, 13, 14, 15, 16, 17, 18, 19, 20, 21,
 22, 23, 24, 25, 26, 27, 58, 59, 60, 61]
```

每个 state/action mapping 都记录 source key、source index、canonical index、semantic、
confidence 和 `alignment_safe`。reserved `64..79` 始终 invalid；其他未激活槽位也必须由 mask
关闭，不能把 canonical 数值 0 当作有效数据。

## 6. 单位和 alignment

joint、head 和 waist 的 position feedback/target 可直接比较，当前 18 个安全槽位得到：

```text
alignment score = 1.0
best lag frames = 0
```

gripper feedback 当前约为毫米范围，command 是 `[0,1]` 闭合目标。这两个 slot 均可作为各自
有明确语义的 state/action 进入模型，但设为 `alignment_safe=false`，不计算错误的同量纲相关性。
normalization 仍分别计算 state 和 action 的 mean/std，不共享统计量。

近静止关节的编码器微噪声可能使纯 MAD 阈值退化到约 `1e-6`。清洗策略为 joint position
增加 `0.01 rad/frame` 的 soft abrupt floor；hard interval 仍是该阈值的 10 倍，即至少
`0.1 rad/frame`。这消除微噪声误报，同时保留对真实大跳变的阻断。

## 7. 预处理命令

全量文件完整后执行：

```bash
export FASTWAM_DATA_ROOT=/path/to/robot_dataset

python3 scripts/run_pipeline.py \
  --datasets agibot_beta \
  --output-root work/stage_pipeline \
  --num-frames 81 \
  --target-fps 20 \
  --verify-files \
  --check-videos
```

产物：

```text
work/stage_pipeline/agibot_beta/
├── scan/episodes.jsonl
├── clean/episodes.cleaned.jsonl
├── canonical/canonical_episodes.jsonl
├── canonical/*.parquet
├── windows/windows.jsonl
├── cases/training_cases.jsonl
└── pipeline_summary.json
```

`--check-videos` 只做稀疏视觉审计。冻结正式 manifest 前再对入选 episode 使用
`--decode-videos` 做完整解码；不要在下载高峰同时对所有 tar 执行 full decode。

## 8. Normalization stats

只允许 train split、A tier、`joint_video_action` 进入 action/state stats：

```bash
python3 scripts/build_fastwam_normalization_stats.py \
  --pipeline-root work/stage_pipeline \
  --datasets all \
  --data-root . \
  --output work/stage_pipeline/normalization_stats.json
```

统计按 `embodiment.normalization_domain + canonical slot` 分组。同一 episode 的多个已准入
start range 先合并，再只统计这些 81/80 窗口覆盖到的唯一行；窗口重叠不会重复计数，已降级
窗口和 episode 尾部未被训练采样的行也不会混回统计。当前 AgiBot validation domain 的
18 个窗口覆盖 761 个唯一 state 行和 760 个唯一 action 行，20 个有效槽位的 count 均为
`761/760`，inactive slot count 为 0，std floor 为 `1e-3`。正式预训练必须在全量 train
manifest 上重建，不能复用该单 episode validation 文件。

## 9. FastWAM 数据配置

FastWAM 侧配置为：

```text
configs/data/canonical_agibot_memory_81f.yaml
configs/task/stage2_agibot_memory_smoke.yaml
```

关键约束：

```yaml
allowed_modes: [joint_video_action]
allowed_quality_tiers: [A]
include_robot_supervision: true
action_dim: 80
proprio_dim: 80
camera_roles: [global_primary, left_wrist, right_wrist]
memory_history_long: 8
memory_history_mid: 2
memory_history_short: 1
```

text embedding 必须在 trainer 启动前预计算，避免每个 worker 加载约 11 GB UMT5：

```bash
cd /path/to/FastWAM
export FASTWAM_PREPROCESS_ROOT=/path/to/Process_WM_Data
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/local/model/checkpoints

python3 scripts/precompute_text_embeds.py \
  task=stage2_agibot_memory_smoke \
  +overwrite=false
```

## 10. 数据与 memory 验证

从数据处理仓库执行：

```bash
python3 scripts/validate_fastwam_training_cases.py \
  --fastwam-repo /path/to/FastWAM \
  --preprocess-root . \
  --datasets agibot_beta \
  --fail-fast
```

验证器使用统一 `work/stage_pipeline/normalization_stats.json`，不再把 robot supervision
硬编码为 RoboCOIN。它检查固定 shape、finite、active slots、A/B loss contract、三相机 mask，
并要求所有有效 memory index 严格小于 `window_start`。当前选择 `window_start=40`，memory
index 为 `29..39`，11/11 有效。

## 11. 单步 optimizer/checkpoint smoke

Stage 2 必须从 Stage 1 video checkpoint 和兼容的 ActionDiT 初始化，不能随机启动：

```bash
cd /path/to/FastWAM
export FASTWAM_PREPROCESS_ROOT=/path/to/Process_WM_Data
export FASTWAM_STAGE1_VIDEO_CKPT=/path/to/stage1/checkpoints/weights/step_N.pt
export FASTWAM_ACTION_DIT_PATH=/path/to/ActionDiT_checkpoint.pt

CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py \
  task=stage2_agibot_memory_smoke
```

smoke 的验收条件：

1. 恰好一个 `window_start=40` 的 A-tier sample。
2. `action_loss_mask=true`，action/state active dimensions 都为 20。
3. `memory_valid_ratio=1.0`，不存在当前/未来帧泄漏。
4. video loss 和 action loss 均 finite。
5. 产生 step 1 轻量权重及 Accelerate optimizer/scheduler/RNG state。
6. 从 state 目录 resume 后能继续 step 2，sampler 不重复或跳过样本。

本次数据、归一化、文本、三相机和 memory loader 已全部通过。当前执行环境
`torch.cuda.is_available() == false` 且 NVML 初始化失败，同时机器上的 8 张 H200 正被另一训练
占用，所以没有抢占设备启动 5B optimizer。配置已经通过 Hydra composition；GPU 可见后只需
运行上述单步命令。

## 12. 全量冻结前检查

1. 七个 proprio tar 和所有 task_info 的 receipt 为 `status=ok`。
2. observation/proprio/task join 覆盖率按 task、episode range 和失败原因汇总。
3. 每个 action index 集合非空、单调、无越界，并记录交集损失比例。
4. 每个 embodiment/domain 报告 active slot、单位、p01/p50/p99、std floor 命中率。
5. 报告 abrupt hard interval、alignment score/lag 和窗口降级率。
6. 三相机 frame count、FPS、duration 和 PTS drift 通过抽样及完整解码校验。
7. normalization 只由冻结后的 train split 计算；validation/test 不贡献统计量。
8. 先完成单库 optimizer/checkpoint/resume，再加入七库 balanced Stage 2。

原始 tar、canonical Parquet、text cache、视频 cache、训练 checkpoint 和 token 都位于忽略的
`work/` 或外部数据目录，不进入公开代码仓库。

## 13. 实现与测试索引

| 功能 | 代码位置 |
|---|---|
| observation/proprio/task join 和 tar offset 索引 | `fastwam_preprocess/adapters/agibot.py` |
| HDF5 valid-index 交集和原始 frame index 保留 | `fastwam_preprocess/source.py` |
| signal、alignment、局部 bad interval 清洗 | `fastwam_preprocess/cleaning.py` |
| joint micro-noise floor 与 abrupt metrics | `fastwam_preprocess/signal_audit.py` |
| train/A/joint 窗口支持集 normalization | `scripts/build_fastwam_normalization_stats.py` |
| commit-locked 组件下载和 size receipt | `scripts/download_agibot_training_assets.py` |
| FastWAM 三相机、mask 和 causal memory 验收 | `scripts/validate_fastwam_training_cases.py` |
| adapter/tar join 回归 | `tests/test_agibot.py` |
| HDF5 index reader 回归 | `tests/test_source.py` |
| noise floor 和 hard jump 回归 | `tests/test_cleaning.py` |
| stats 过滤、窗口并集与去重回归 | `tests/test_normalization_stats.py` |
