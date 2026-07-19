# FastWAM、old LingBot-VA 与 DreamZero 统一微调

[文档索引](../README.md)

更新日期：2026-07-19

本文说明如何在 `Process_WM_Data` 中用同一份机器人源数据准备并启动三个模型。模型代码、
权重、运行缓存和 checkpoint 都留在仓库外或 `work/`；Git 仓库只保存 target 生成器、运行时
适配、命令构造器、配置模板和测试。

## 1. 边界和目录

三个模型共享 source identity、episode split、语言和物理相机语义，但不共享最终张量或
normalization：

| 模型 | Target | 时间合同 | Action 合同 |
|---|---|---|---|
| Memory FastWAM | `TrainingCaseV1` | 20 Hz，81 state / 80 action / 21 video | canonical 80D + valid mask |
| old `Robbyant/lingbot-va` | LeRobot v2 + latent | 源 28 Hz，VAE 输入 7 Hz，latent 1.75 Hz | compact 8D 映射到模型 30D channel |
| DreamZero | GEAR LeRobot + Hydra profile | 28 Hz，33 video / 24 action | XDOF right joint 7D + gripper 1D |

代码布局：

```text
fastwam_preprocess/       source scan、清洗、canonical 和 TrainingCaseV1
targets/lingbot_va/       old LingBot target、latent 提取和 import shim
targets/dreamzero/        GEAR target、Hydra profile 和安装校验
tuning/                   三模型命令构造、收据、状态查询
scripts/tune_models.py    统一 dry-run / run / status 入口
scripts/run_*_training.py 不修改外部仓库的运行时适配
configs/tuning/           可迁移配置模板
work/                     数据、cache、日志和 checkpoint，Git 忽略
```

## 2. 自采数据的固定语义

当前 `take_wrong_item_right_arm` 是 LeRobot v2.1，44 episodes、31,359 frames、28 Hz，原始
state/action 均为 15D。右臂监督固定使用：

```text
right arm joints: source 0..6
right gripper:    source 14
```

采集时四路相机线接反，因此训练不能按 key 名猜物理位置。当前固定绑定为：

| 数据 key | 实际画面 | 三模型用途 |
|---|---|---|
| `observation.images.left_eye` | 头部/顶部相机 | global/head view |
| `observation.images.right_eye` | 左腕相机 | left wrist view |
| `observation.images.right_wrist` | 右腕相机 | right wrist view |
| `observation.images.left_wrist` | 额外/不确定视角 | 当前不进主训练输入 |

推理必须使用同一物理绑定。后续修正硬件或 key 时，应新增有版本的 camera profile，不能静默
改旧数据含义。

## 3. Target 准备

### 3.1 FastWAM

```bash
python3 scripts/run_pipeline.py \
  --datasets lingbot_va \
  --input-root lingbot_va=/data/take_wrong_item_right_arm \
  --output-root work/tuning/fastwam_data \
  --num-frames 81 \
  --target-fps 20 \
  --verify-files \
  --check-videos

python3 scripts/build_fastwam_normalization_stats.py \
  --pipeline-root work/tuning/fastwam_data \
  --datasets lingbot_va \
  --data-root . \
  --output work/tuning/fastwam_data/normalization_stats.json
```

只允许 `train + A + joint_video_action` 进入 action stats。坏区间可降级为 `video_only`，但不得
用零 action 冒充监督。

### 3.2 old LingBot-VA

```bash
python3 scripts/prepare_lingbot_va_target.py prepare \
  --source-root /data/take_wrong_item_right_arm \
  --output-root work/tuning/targets/take_wrong_item_lingbot_va \
  --profile take_wrong_item_right_arm \
  --verify-files

/path/to/lingbot/python scripts/extract_lingbot_va_latents.py \
  --target-root work/tuning/targets/take_wrong_item_lingbot_va \
  --model-root /models/lingbot-va-base \
  --lingbot-repo /code/lingbot-va \
  --episode 0 \
  --max-segments 1 \
  --device cuda
```

Wan causal VAE 使用首帧单独编码、其后每四帧一组的 `1 + 4 + 4 + ...` 分块。固定四帧分块会
产生 temporal residual shape mismatch。旧 LingBot 的 `datasets` 版本也不支持新版 Arrow
metadata 的 `List` feature；target 物化仅移除这层 producer metadata，列类型和值不变。

`--episode 0 --max-segments 1` 只用于 smoke。正式训练前必须不带这两个限制完成 132 个三视角
latent job，并执行：

```bash
python3 scripts/prepare_lingbot_va_target.py validate \
  --root work/tuning/targets/take_wrong_item_lingbot_va \
  --require-latents \
  --verify-files
```

统一 launcher 默认拒绝 latent 不完整的 target。只有局部 smoke 配置才可显式设置
`models.lingbot_va.allow_partial_latents: true`；即使开启，也必须至少有一个 segment 的三路
latent 全部存在。仓库中的可迁移示例保持 `false`。

### 3.3 DreamZero

```bash
python3 scripts/prepare_dreamzero_target.py prepare \
  --source-root /data/take_wrong_item_right_arm \
  --output-root work/tuning/targets/take_wrong_item_dreamzero \
  --profile take_wrong_item_right_arm \
  --verify-files

python3 scripts/install_dreamzero_profile.py \
  --target-root work/tuning/targets/take_wrong_item_dreamzero \
  --dreamzero-repo /code/dreamzero
```

安装器校验 `EmbodimentTag.XDOF` 和 projector index，不会修改模型代码。生成 profile 必须带
`# @package _global_`，否则 Hydra 会把 `max_chunk_size` 等参数错误挂到 `data.*`。

## 4. 统一运行配置

复制环境变量模板的结构，实际路径可写到 Git 忽略的本机 YAML：

```text
configs/tuning/take_wrong_item.example.yaml
work/tuning/take_wrong_item.local.yaml
```

先检查命令，不启动模型：

```bash
python3 scripts/tune_models.py dry-run \
  --config work/tuning/take_wrong_item.local.yaml \
  --model fastwam \
  --phase stage3_finetune \
  --output-dir work/tuning/runs/fastwam_stage3 \
  --steps 1 \
  --gpus 0
```

`dry-run` 会验证所有声明为 file/directory 的输入，并输出结构化 argv。runner 不使用 shell
拼接，因此路径和 Hydra list 不会被二次解释。

## 5. 三模型启动

### 5.1 FastWAM 三阶段

```bash
# Stage 1: video backbone
python3 scripts/tune_models.py run --config "$CFG" \
  --model fastwam --phase stage1_video \
  --output-dir work/tuning/runs/fastwam_stage1 --steps 1 --gpus 0

# Stage 2: MemoryFastWAM
python3 scripts/tune_models.py run --config "$CFG" \
  --model fastwam --phase stage2_memory \
  --output-dir work/tuning/runs/fastwam_stage2 --steps 1 --gpus 0

# Stage 3: 当前任务/本体微调
python3 scripts/tune_models.py run --config "$CFG" \
  --model fastwam --phase stage3_finetune \
  --output-dir work/tuning/runs/fastwam_stage3 --steps 1 --gpus 0
```

天机自采数据的固定单窗口过拟合、memory-aware 想象视频和 GT 三联 demo 使用独立的
`overfit` phase，详见 [FastWAM 天机单轨迹过拟合](FASTWAM_TIANJI_OVERFIT.md)。它不会复用
Stage 3 的随机样本评测，也不会每个 step 保存完整 optimizer state。

跨阶段用轻量 `.pt` 初始化；同阶段恢复必须用 `checkpoints/state/step_N`，它包含 optimizer、
scheduler、sampler 和 RNG。

### 5.2 old LingBot-VA

```bash
python3 scripts/tune_models.py run --config "$CFG" \
  --model lingbot_va --phase finetune \
  --output-dir work/tuning/runs/lingbot_va --steps 250 \
  --gpus 0,1,2,3,4,5,6,7
```

wrapper 处理三个上游问题：缺少 `flash_attn` 时用 PyTorch SDPA 提供 import-compatible fallback；
单 target 直接使用官方 `LatentLeRobotDataset`，避免模型/NCCL 初始化后 fork 固定 128 个进程；
checkpoint 补存 optimizer、scheduler 和 step。

整条 episode 长度不同，`batch_size>1` 时必须启用同步固定窗口。当前 8 卡 H200 全量 overfit
配置为 `batch_size=24/GPU`、`window_frames=16`、`samples_per_episode=48`，global batch 为 192；
44 条 episode 组成 2,112 个虚拟窗口样本，每 rank 264 个样本，恰好 11 个整 batch。窗口起点
在每次读取时重新随机采样，video latent、action 和 mask 使用同一裁剪区间。

### 5.3 DreamZero

```bash
python3 scripts/tune_models.py run --config "$CFG" \
  --model dreamzero --phase finetune \
  --output-dir work/tuning/runs/dreamzero --steps 500 \
  --gpus 0,1,2,3,4,5,6,7
```

当前全量 overfit 配置为 8 卡、`batch_size=1/GPU`、global batch 8、500 steps，保存点为
125/250/375/500。DreamZero 的每卡完整模型常驻约 85.6 GB，首个 forward 后峰值约 118.7 GB，
因此 H200 上不把 per-GPU batch 提到 2。`ATTENTION_BACKEND=torch` 且不启用 DeepSpeed。
首次运行会加载 10 个 DreamZero shard、缓存数据并编译部分算子；这是冷启动成本，不是数据卡死。

上游 `BaseExperiment` 在 Trainer 已按 `save_lora_only=true` 写完 step checkpoint 后，仍会调用
generic saver 导出完整 16.5B 模型。统一 wrapper 跳过这份重复的最终全模型；可恢复的 LoRA、
optimizer、scheduler、RNG 和 trainer state 均保留在 `checkpoint-N/`。

训练完成后的 8-case GT-observation Pair 推理默认生成 913 帧/case；这是当前最长 8 条合格
episode 的最大统一 `1 + 8k` horizon，81 帧只保留为最低协议门槛。正式 step-500 推理已完成
8×913 帧和 7304 帧 reel，33 个 MP4 共 36,520 帧实际解码通过。长序列通过 24-chunk 有界
KV/VAE context 分段执行，见
[DreamZero Target](../targets/DREAMZERO.md#gt-observation-pair-推理)。

DreamZero 最小环境除官方核心依赖外还需要：

```bash
pip install h5py matplotlib termcolor lark lmdb msgpack msgpack-numpy
pip install 'albumentations==1.4.18' \
  'numpy==1.26.4' 'opencv-python-headless==4.10.0.84'
```

必须在安装 Albumentations 后再次固定 NumPy/OpenCV，避免 pip 选择要求 NumPy 2.x 的 OpenCV 5。

## 6. 断点恢复

FastWAM：

```bash
python3 scripts/tune_models.py run --config "$CFG" \
  --model fastwam --phase stage3_finetune \
  --output-dir work/tuning/runs/fastwam_stage3 --steps 2 --gpus 0 \
  --resume work/tuning/runs/fastwam_stage3/checkpoints/state/step_000001
```

LingBot：

```bash
python3 scripts/tune_models.py run --config "$CFG" \
  --model lingbot_va --phase finetune \
  --output-dir work/tuning/runs/lingbot_va --steps 2 --gpus 1 \
  --resume work/tuning/runs/lingbot_va/checkpoints/checkpoint_step_1
```

DreamZero：

```bash
python3 scripts/tune_models.py run --config "$CFG" \
  --model dreamzero --phase finetune \
  --output-dir work/tuning/runs/dreamzero --steps 2 --gpus 2 \
  --resume work/tuning/runs/dreamzero/checkpoint-1
```

DreamZero wrapper 显式传递 checkpoint，避免 output root 已有最终 `config.json` 时上游把任务判定为
“训练已结束”而跳过恢复。

## 7. 日志、收据和状态

每次运行写入：

```text
<output>/_wm_tuning/<UTC-run-id>.log
<output>/_wm_tuning/<UTC-run-id>.json
```

收据包含 argv、受控环境变量、外部仓库 SHA/dirty 状态、开始结束时间、退出码和发现的
checkpoint，不记录 token 或完整进程环境。

```bash
python3 scripts/tune_models.py status --config "$CFG" \
  --model fastwam --output-dir work/tuning/runs/fastwam_stage3
```

## 8. 完成定义

每个模型的 `Training-validated` 至少要求：

1. 官方 dataset/collator 能读取真实 target sample，shape 和有限值通过。
2. 完成一次真实 forward、backward 和 optimizer step，loss 有限。
3. checkpoint 写出且非空。
4. 从该 checkpoint 恢复并完成下一个 optimizer step。
5. 收据记录外部 SHA 和成功退出码。

单 episode、单 segment 或单 step 只证明 smoke path，不代表全量 latent、长程收敛、多 GPU、
闭环推理或生产训练已经完成。当前实测结论统一维护在
[Validation Status](../reference/VALIDATION_STATUS.md)。
