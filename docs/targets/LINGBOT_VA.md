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
- `take_wrong_item_right_arm`：从 15D 中选择 `[0:7] + [14]`，映射到 right joints `21..27` 和 right gripper `29`。源数据 28 Hz 先采样到 7 Hz，Wan VAE 再按 4 帧压缩，因此每个 latent 对齐 `28 / 7 * 4 = 16` 个 action。

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

Wan causal VAE 使用首帧产生第一个 latent，之后每个完整 4 帧块产生一个 latent。job 的采样帧数
因此固定为 `1 + 4k`；episode 尾部不足 4 帧的残块会被丢弃。这样 VAE 的
`latent_num_frames`、loader 的 action reshape 和 server 的 `action_per_frame` 使用同一时间轴。
抽取器也会对旧 manifest 执行相同的防御性截断。

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

整条 episode 的 latent 时间长度不同，默认 collate 只支持 `batch_size=1`。大 batch 训练应设置
固定 latent 窗口，wrapper 会用同一 `[start:end]` 同步裁剪 video latent、action 和 action mask：

```yaml
batch_size: 24            # 每 GPU
window_frames: 16         # 约 9.14 秒：16 / (7 / 4)
samples_per_episode: 48   # 44 * 48 / 8 GPU / 24 = 11 个整 batch
```

当前自采 target 已验证 44 episodes、132/132 三视角 latent，窗口训练使用全部 episode，不是单条
轨迹复制。8 张 H200 上 `batch_size=24/GPU` 的稳定实测峰值为 103.03 GiB allocated、
120.25 GiB reserved；`batch_size=32` 不保留足够 allocator 余量。

## Observation-conditioned GT Pair 推理

`scripts/run_lingbot_va_pair_inference.py` 用于 overfit 后的多 case 可视化评测。默认的
`gt_chunk_feedback` 模式复现官方在线 client 时序：首次预测输入起始三相机 GT observation；执行完
每个 chunk 后，把该段环境返回的多相机 key frames 通过 `_compute_kv_cache()` 写回，再预测下一
chunk。离线 replay 没有真实环境，因此环境 observation 取自 GT 轨迹，action history 则默认使用
模型上一段 prediction，与官方 client 的 `state=action` 一致。

当前 chunk 只能看到前一边界及更早的 observation，不会看到正在评测的未来 GT。GT latent 历史
还会作为 causal VAE 的 decode context，避免分段解码产生亮度跳变；它只参与像素解码，不会注入
transformer 的未来上下文。

```bash
/path/to/lingbot/python scripts/run_lingbot_va_pair_inference.py \
  --lingbot-repo /code/lingbot-va \
  --dataset-root /data/targets/take_wrong_item_lingbot_va \
  --base-model-root /models/lingbot-va-base \
  --checkpoint /runs/lingbot_va/checkpoints/checkpoint_step_250 \
  --output-dir /runs/lingbot_va/inference_step_250 \
  --num-cases 8 \
  --num-chunks 10 \
  --observation-mode gt_chunk_feedback \
  --feedback-action-source predicted \
  --gpus 0,1,2,3,4,5,6,7
```

`--observation-mode open_loop` 只注入初始 observation，用作消融对照；不应作为 LingBot-VA
在线策略的主评测。`--feedback-action-source gt` 会同时 teacher-force action history，仅适合诊断。

默认选择长度足够的 8 条最长 episode，每张 GPU 独立运行一个 case。也可用
`--episodes 7,8,32,3,2,20,22,10` 固定集合。当前 profile 下每个 case 的时间轴为：

```text
10 chunks * 4 latent/chunk = 40 latent frames
Wan causal decode: 1 + 4 * (40 - 1) = 157 video frames
comparison FPS: 7
duration: 157 / 7 = 22.43 seconds
GT source IDs: 0, 4, ..., 624 at source 28 FPS
GT observation feedback IDs: 0, 4, ..., 560
useful predicted actions: 3 * 16 + 9 * 4 * 16 = 624 compact 8D steps
```

输出包含：

```text
lingbot_va_8case_gt_vs_imagined_reel.mp4  # 8 case 串联总览
episode_XXXXXX/gt_vs_imagined_all_views.mp4
episode_XXXXXX/gt_vs_imagined_<camera>.mp4
episode_XXXXXX/gt_and_predicted_actions.npz
episode_XXXXXX/receipt.json
inference_receipt.json
INDEX.md
```

总览视频上排是 GT，下排是 imagined，三列固定为 global/head、left wrist、right wrist。
`receipt.json` 明确记录 `gt_feedback_after_initial_observation=true`、每个实际注入的源 frame ID、
`feedback_action_history_source=predicted`、采样参数、相机物理角色、action 误差和逐相机 PSNR。
运行时模型目录只包含指向 base components 与 checkpoint transformer 的符号链接，不复制权重。

## 当前限制

- 输入必须已经是 LeRobot v2；HDF5/RLDS 到 LeRobot 的视频编码不在该 target 中隐式完成。
- 本仓库不提交或下载 Wan2.2 权重，也不伪造 latent。
- 相机 key 顺序直接决定 latent 拼接顺序，修改 profile 后必须重建 receipt 和全部 latent。
- 模型结构和损失仍来自官方 LingBot-VA checkout；本仓库只提供可移植 wrapper 和数据合同。
- `samples_per_episode` 是随机窗口的虚拟采样倍数，不会复制 latent 文件；训练仍从原 44 条 episode 动态裁窗。
