# Old LingBot-VA Target

[Target 输出索引](README.md)

本文对应旧版 [`Robbyant/lingbot-va`](https://github.com/Robbyant/lingbot-va)，不是 `Robbyant/lingbot-vla-v2`。

## 官方合同

当前 target profile 固定到：

```text
repository: https://github.com/Robbyant/lingbot-va
revision: 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb
LeRobot: v2.1 / lerobot 0.3.3 loader contract
model action space: 30D
```

30D 模型 channel：

| Channel | 语义 |
|---:|---|
| `0..6` | left EEF pose |
| `7..13` | right EEF pose |
| `14..20` | left arm joints |
| `21..27` | right arm joints |
| `28` | left gripper |
| `29` | right gripper |

源数据可以是更短的 compact action。`used_action_channel_ids` 描述每个 compact dimension 落到哪个 30D channel，未使用 channel 由官方 loader padding 并 mask。

## 已提供 Profile

`configs/targets/lingbot_va.yaml` 包含：

- `robotwin`：16D 双臂 EEF/gripper，三相机。
- `libero`：7D 单臂 delta EEF/gripper，两相机。
- `take_wrong_item_right_arm`：从 15D 中选择 `[0:7] + [14]`，映射到 right joints `21..27` 和 right gripper `29`。

Profile 同时固定相机顺序、分辨率、latent FPS、frame chunk 和 action-per-frame。不能只按 action width 自动猜语义。

`robotwin` profile 还会在每个 `action_config` segment 内，把左右两组 7D
`xyz + xyzw` EEF pose 转成相对该 segment 首帧的位姿，再计算 q01/q99。这与旧版
LingBot-VA 官方 RoboTwin loader 的 normalization 语义一致；原始 Parquet 仍保留
absolute action，不会被这一步覆盖。

## Prepare

```bash
python3 scripts/prepare_lingbot_va_target.py prepare \
  --source-root /data/lerobot_v2_dataset \
  --output-root /data/targets/lingbot_va_dataset \
  --profile take_wrong_item_right_arm \
  --link-mode symlink \
  --verify-files \
  --train-episodes-file /data/splits/train_episodes.txt
```

`train_episodes.txt` 每行一个 episode index，也可以是 JSON integer list。quantile stats 只使用该列表；不传时使用全部 episodes，仅适合没有 held-out 的 smoke。

输出：

```text
meta/episodes.jsonl                       # 增加/验证 action_config
meta/lingbot_va_model_profile.json        # 30D channel、inverse map、q01/q99
meta/lingbot_va_latent_jobs.jsonl         # 每 segment × camera 的 VAE job
meta/lingbot_va_target_receipt.json       # source/profile/revision/readiness
data/                                     # 链接或 action 压缩后的 Parquet
videos/                                   # 默认链接原始视频
```

如果 profile 的 `source_action_indices` 不是原 action 的完整 identity，prepare 会重写输出 Parquet 的 `action` 列，原始 Parquet 不修改。

## Action Segmentation

已有合法 `action_config` 时原样保留；缺失时为整条 episode 生成一个 segment：

```json
{
  "start_frame": 0,
  "end_frame": 450,
  "action_text": "task description"
}
```

重叠、越界、空文本或空 segment 会 hard fail，不会静默修复。

## VAE Latent Gate

prepare 只生成 deterministic latent jobs，不加载模型权重。每个 job 固定 camera、segment、frame IDs、FPS、文本和官方文件名：

```text
latents/chunk-000/<camera>/episode_000000_0_450.pth
```

Wan2.2 VAE 和 text encoder 必须在 LingBot-VA 模型环境执行。提取完成后运行：

```bash
/path/to/lingbot/python scripts/extract_lingbot_va_latents.py \
  --target-root /data/targets/lingbot_va_dataset \
  --model-root /models/lingbot-va-base \
  --lingbot-repo /code/lingbot-va \
  --device cuda

python3 scripts/prepare_lingbot_va_target.py validate \
  --root /data/targets/lingbot_va_dataset \
  --require-latents \
  --verify-files
```

缺 latent 时 metadata 可以 `valid=true`，但 `ready_for_training` 必须为 `false`。
统一训练 launcher 也默认检查全部 job；`allow_partial_latents: true` 只用于至少有一个
完整多视角 segment 的 smoke。

## 训练入口

`scripts/run_lingbot_va_training.py` 使用官方 `LatentLeRobotDataset` 和 Trainer，补齐了单 target
训练、无 FlashAttention 时的 SDPA import fallback，以及 model/optimizer/scheduler/step 的完整
checkpoint。统一启动和恢复命令见 [三模型统一微调](../training/THREE_MODEL_TUNING.md)。

## 当前限制

- 输入必须已经是 LeRobot v2；HDF5/RLDS 到 LeRobot 的视频编码不在该 target 中隐式完成。
- 本仓库不提交或下载 Wan2.2 权重，也不伪造 latent。
- 相机 key 顺序直接决定 latent 拼接顺序，修改 profile 后必须重建 receipt 和全部 latent。
- 模型结构和损失仍来自官方 LingBot-VA checkout；本仓库只提供可移植 wrapper 和数据合同。
- 当前实测只提取了自采数据 episode 0 的三个 latent；全量训练前还需完成其余 129 个。
