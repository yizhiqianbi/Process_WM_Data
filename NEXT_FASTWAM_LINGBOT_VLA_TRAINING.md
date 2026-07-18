# FastWAM 与 LingBot-VLA 下一阶段训练执行手册

更新日期：2026-07-18

本文是接下来训练工作的统一入口。目标不是再次证明某个 Python 函数能运行，而是把数据、模型、checkpoint、恢复、评测和真机前检查连成两条可复现链路：

1. 基于九个逻辑数据集完成 FastWAM 的三阶段训练。
2. 复现并完善 `LingBot-VLA v2` 在自采任务上的后训练。

当前范围不包含 Cosmos。

## 1. 名词边界

以下两个名字不能混用：

- **LingBot-VLA v2**：`Robbyant/lingbot-vla-v2` 模型及其 6B checkpoint。对应本地代码仓库 `LingBot-VLA-v2-Custom-Finetune`，已经在 `jokeru/take_wrong_item_right_arm` 上完成 2000-step 微调。
- **LingBot-VA**：FastWAM 数据管线中的上游数据来源，包括 RoboTwin cleaned/augmented 和 LIBERO-Long。它在本仓库中对应 `lingbot_va` adapter，不是 LingBot-VLA v2 的训练代码。

本文中的“LingBot 训练”默认指 **LingBot-VLA v2**。若后续目标改为旧版 `Robbyant/lingbot-va` 模型训练，必须单独建立环境、配置和 checkpoint 合同，不能复用本文的 LingBot-VLA 命令。

## 2. 当前真实状态

| 工作流 | 已完成 | 尚未完成 |
|---|---|---|
| Process_WM_Data | 九库 adapter、81/80/21 TrainingCaseV1、A/B/C admission、样本级真实验证 | 九库全量 pipeline、全量视频验收、全量 train-only stats |
| FastWAM Stage 1 | 原七库逐库 5B smoke、balanced sampler、checkpoint | 包含 LingBot-VA/DreamZero 的九库回归；正式长训 |
| FastWAM Stage 2 | 8/2/1 memory、因果 mask、原七库 smoke、checkpoint | 九库 stats 后联合 optimizer/resume；正式 memory pretrain |
| FastWAM Stage 3 | RoboCOIN 单步微调、strict Stage 2 恢复、memory inference | 独立 validation、完整微调、rollout |
| LingBot-VLA v2 | 44 episodes 全量处理、2-step/100-step smoke、2000-step 8-GPU 微调、open-loop replay | episode-level held-out 训练；真机 shadow/closed-loop |

已经得到的 LingBot-VLA v2 基线：

```text
training: 2000/2000, no NaN/Inf, no ignored batch, no rank failure
step 1500 replay: MSE 0.008946, MAE 0.057300
step 2000 replay: MSE 0.007354, MAE 0.051602
candidate: global_step_2000/hf_ckpt
```

这些 replay episode 参与过训练，只能说明拟合和推理链路正常，不能作为泛化或真机成功率。

## 3. 两条训练链不能共享什么

两套模型可以读取相同的 raw robot data，但训练合同不同：

| 项目 | FastWAM Memory | LingBot-VLA v2 |
|---|---|---|
| 数据入口 | `TrainingCaseV1` + canonical sidecar | LeRobot v3.0 |
| 时间合同 | 81 state / 80 action / 21 video | 50-step action chunk |
| Action 空间 | 80D canonical + dimension mask | 55D unified padding + joint mask |
| 视频布局 | 3-camera composite `[3,21,384,320]` | 三路独立 Qwen3-VL 图像输入 |
| 语言 | 预计算 UMT5 `[128,4096]` | Qwen3-VL tokenizer |
| Checkpoint | Stage-specific `.pt` + Accelerate state | DCP state + HF safetensors |

因此禁止：

- 把 FastWAM normalization stats 交给 LingBot-VLA。
- 把 LingBot-VLA 的 `hf_ckpt` 作为 FastWAM Stage 2 初始化。
- 为了“统一格式”删除 mask、相机角色或源数据 provenance。
- 在 action 语义未确认时把 video-only case 升级成 action supervision。

## 4. 代码与目录

当前机器：

```bash
export ROBOT_DATA_ROOT=/public/interns/hubin/dataset/robot_dataset
export PREPROCESS_REPO=$ROBOT_DATA_ROOT/Preprocess_FastWAM
export FASTWAM_REPO=/public/interns/hubin/code/FastWAM
export LINGBOT_REPO=/public/interns/hubin/code/LingBot-VLA-v2-Custom-Finetune
```

可迁移到其他服务器的代码仓库：

```text
https://github.com/yizhiqianbi/Process_WM_Data
https://github.com/yizhiqianbi/LingBot-VLA-v2-Custom-Finetune
```

FastWAM 当前工作树包含 memory、canonical loader、sampler 和三阶段配置修改，且 `origin` 仍指向上游 `yuantianyuan01/FastWAM`。正式训练前必须先完成以下 P0 工作：

1. 创建自己的 FastWAM fork。
2. 审计当前 dirty diff，不覆盖已有修改。
3. 提交代码、配置和 tests，不提交 `checkpoints/`、数据或 `work/`。
4. 记录 commit SHA，并用该 SHA 重新执行三阶段 smoke。

不允许从未提交的 FastWAM 工作树直接启动数天的生产训练，否则 checkpoint 无法可靠复现。

## 5. 统一实验记录

每次训练必须使用新的输出目录：

```text
runs/<model>/<stage>/<dataset-or-mixture>/<UTC-date>_<git-sha>/
```

至少保存：

```text
command.txt
resolved_config.yaml
code_commit.txt
data_manifest.json
normalization_manifest.json
environment.txt
train.log
metrics.jsonl
checkpoints/
evaluation/
```

`data_manifest.json` 必须记录 dataset revision、case manifest SHA256、train/validation episode 数量、窗口数量、normalization domain 和 camera-role 分布。仅保存 YAML 不足以复现实验。

## 6. FastWAM 执行顺序

### 6.1 环境和基础权重

建议独立 Python 3.10 环境：

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
python -m pip install -U pip
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
cd "$FASTWAM_REPO"
python -m pip install -e .
```

设置模型路径：

```bash
export FASTWAM_PREPROCESS_ROOT="$PREPROCESS_REPO"
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/wan22/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_DIT_PATH="$DIFFSYNTH_MODEL_BASE_PATH/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
```

硬门槛：Wan2.2-TI2V-5B shards、VAE、T5/tokenizer 和 ActionDiT 均可本地读取；训练节点不应在 rank 启动后临时联网下载权重。

### 6.2 全量数据门槛

先确认下载状态和固定 revision：

```bash
cd "$PREPROCESS_REPO"
python scripts/download_datasets.py status \
  --datasets all \
  --data-root "$ROBOT_DATA_ROOT" \
  --output "$ROBOT_DATA_ROOT/download_status.json"

python scripts/download_datasets.py verify \
  --datasets all \
  --data-root "$ROBOT_DATA_ROOT" \
  --workers 4 \
  --output "$ROBOT_DATA_ROOT/download_verify.json"
```

当前没有证据表明 LingBot-VA 和 DreamZero 已在主数据目录完成全量下载。它们的 locked metadata 规模约为 208 GiB、4.4 GiB 和 33 GiB，必须先下载并生成成功 receipt。

第一轮全量 pipeline 不做 full decode：

```bash
export FASTWAM_DATA_ROOT="$ROBOT_DATA_ROOT"
python scripts/run_pipeline.py \
  --datasets all \
  --output-root work/stage_pipeline \
  --num-frames 81 \
  --target-fps 20 \
  --workers 2 \
  --verify-files
```

完成结构、action admission 和 manifest 审计后，再对 A/B candidate 运行视觉检查：

```bash
python scripts/run_pipeline.py \
  --datasets all \
  --output-root work/stage_pipeline \
  --num-frames 81 \
  --target-fps 20 \
  --workers 2 \
  --verify-files \
  --check-videos \
  --decode-videos
```

全量验收：

```bash
python scripts/validate_fastwam_training_cases.py

python scripts/build_fastwam_normalization_stats.py \
  --pipeline-root work/stage_pipeline \
  --datasets all \
  --data-root . \
  --output work/stage_pipeline/normalization_stats.json
```

只有 `train + A + joint_video_action` case 进入 action/state stats。B tier 只参与 video loss，不查询动作统计。

### 6.3 文本与 tar cache

先预热 TrainingCase 引用的 tar 视频 member，再生成全部唯一 prompt 的 UMT5 cache：

```bash
cd "$FASTWAM_REPO"
python scripts/prewarm_tar_video_cache.py \
  --case-manifest "$PREPROCESS_REPO/work/stage_pipeline/agibot_beta/cases/training_cases.jsonl" \
  --case-manifest "$PREPROCESS_REPO/work/stage_pipeline/galaxea/cases/training_cases.jsonl" \
  --data-root "$PREPROCESS_REPO" \
  --cache-dir "$PREPROCESS_REPO/work/stage_pipeline/video_member_cache" \
  --workers 3

CUDA_VISIBLE_DEVICES=0 python scripts/precompute_text_embeds.py \
  task=stage1_video_backbone_pretrain +overwrite=false
```

当前 FastWAM 的 `canonical_stage1_all.yaml` 和 `canonical_stage2_memory_all.yaml` 仍只列原七库。
LingBot-VA/DreamZero 全量 case 通过后，必须先把下面两个 manifest 显式加入 Stage 1/2 data
config，再生成 text cache：

```text
work/stage_pipeline/lingbot_va/cases/training_cases.jsonl
work/stage_pipeline/dreamzero/cases/training_cases.jsonl
```

若不更新配置，`stage1_all_datasets_*` 和 `stage2_all_datasets_*` 仍然只是七库运行，不能在报告中称为九库训练。cache 和 text embeds 必须有 manifest；不能仅依赖目录“看起来存在”。

### 6.4 Stage 1: video backbone

加入并审计 LingBot-VA/DreamZero manifests 后，再执行九库单步/逐库 smoke：

```bash
cd "$FASTWAM_REPO"
CUDA_VISIBLE_DEVICES=0 python scripts/train.py task=stage1_video_backbone_smoke
CUDA_VISIBLE_DEVICES=0 python scripts/train.py task=stage1_all_datasets_smoke
```

在配置更新前，上述 `stage1_all_datasets_smoke` 的已知基线是原七库 7/7，不是九库。

Stage 1 smoke 验收：

- 每个纳入 mixture 的数据集至少真实解码一个非首窗口。
- loss finite，且只有 video DiT 参数产生梯度。
- checkpoint 包含 `dit`，可被 Stage 2 strict load。
- 保存后可从 Accelerate state 恢复并继续至少一个 optimizer step。

多 GPU 正式训练前，先用 2 GPU 验证 sampler、梯度累积和 resume。通过后启动生产配置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python scripts/train.py \
  task=stage1_video_backbone_pretrain \
  output_dir="$PREPROCESS_REPO/work/stage_pipeline/runs/stage1_video_pretrain_<run-id>"
```

配置中的 `100000` steps 是预算上限，不是默认最优点。先保存 1k/5k/10k checkpoint，并根据 validation video loss、数据重复率和生成质量决定是否继续。

### 6.5 Stage 2: MemoryFastWAM

Stage 2 必须显式加载 Stage 1 video checkpoint：

```bash
export FASTWAM_STAGE1_VIDEO_CKPT=/path/to/stage1/checkpoints/weights/step_N.pt
test -f "$FASTWAM_STAGE1_VIDEO_CKPT"
test -f "$FASTWAM_ACTION_DIT_PATH"
```

先执行 memory smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py task=stage2_memory_active_smoke
CUDA_VISIBLE_DEVICES=0 python scripts/train.py task=stage2_all_datasets_memory_smoke
```

同样，必须确认 resolved data config 实际包含 9 个 case manifests；当前已完成的基线是原七库。

Stage 2 验收：

- current video 为 21 帧，history 严格来自窗口起点之前。
- 非首窗口 `memory_valid_ratio=1.0`，memory token 数为 168。
- A tier 同时产生 finite video/action loss。
- B tier 的 `action_loss_mask=false` 且 action loss 严格为 0。
- checkpoint 包含 `mot`、`memory_patchers`、`proprio_encoder`。
- 同阶段 state resume 和跨阶段轻量 `.pt` 初始化均通过。

生产训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python scripts/train.py \
  task=stage2_memory_fastwam_pretrain \
  output_dir="$PREPROCESS_REPO/work/stage_pipeline/runs/stage2_memory_pretrain_<run-id>"
```

不得在未设置 `FASTWAM_STAGE1_VIDEO_CKPT` 时启动 Stage 2，也不得借用当前只覆盖 RoboCOIN/AgiBot 的小样本 stats 进行九库正式训练。

### 6.6 Stage 3: 专项微调

第一条闭环建议继续使用已验证的 RoboCOIN：

```bash
export FASTWAM_STAGE2_MEMORY_CKPT=/path/to/stage2/checkpoints/weights/step_N.pt

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  task=stage3_robocoin_memory_smoke \
  output_dir="$PREPROCESS_REPO/work/stage_pipeline/runs/stage3_robocoin_smoke_<run-id>"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/train.py \
  task=stage3_robocoin_memory_finetune \
  output_dir="$PREPROCESS_REPO/work/stage_pipeline/runs/stage3_robocoin_<run-id>"
```

Stage 3 必须从 Stage 2 strict 恢复 8/2/1 memory 合同。至少比较：

- Stage 2 zero-shot。
- Stage 3 finetuned。
- Stage 3 no-memory ablation。
- 不同 memory 历史长度。
- 不同 `window_start`，包含 episode 早期 padding 场景。

最终指标必须包含反归一化 action error 和 rollout success，不能只报告 diffusion training loss。

## 7. LingBot-VLA v2 执行顺序

### 7.1 代码和环境

公开代码：

```bash
git clone git@github.com:yizhiqianbi/LingBot-VLA-v2-Custom-Finetune.git
cd LingBot-VLA-v2-Custom-Finetune
scripts/bootstrap_upstream.sh
```

固定 upstream、Python 3.12、Torch 2.8 和 LeRobot 0.4.2。模型、数据、token 和输出目录必须位于代码仓库之外。以 `.env.example` 为模板设置路径，然后导出：

```bash
set -a
source .env
set +a
```

安装与已验证环境一致的依赖：

```bash
"$LINGBOT_PYTHON" -m pip install -r .upstream/lingbot-vla-v2/requirements.txt
"$LINGBOT_PYTHON" -m pip install --no-deps lerobot==0.4.2
"$LINGBOT_PYTHON" -m pip install -e .
```

这里对 LeRobot 使用 `--no-deps`，避免其发布依赖范围改写 LingBot 固定的 Torch 和 datasets 版本。

### 7.2 数据链路

```bash
scripts/download_dataset.sh
scripts/validate_dataset.sh --decode-videos
scripts/prepare_dataset.sh

# 数据所有者确认 15D state/action 和错线相机的实际画面角色后执行
scripts/validate_dataset.sh --decode-videos --accept-inferred-layout

scripts/render_configs.sh
scripts/compute_norm_stats.sh
scripts/smoke_loader.sh
scripts/smoke_full_sample.sh --index 0
```

验收基线：44 episodes、31,359 frames、三路训练相机、8 个 active action/state dims、50-step action chunk，所有 tensor finite。

### 7.3 GPU smoke 与正式训练

```bash
scripts/check_environment.sh --require-cuda
scripts/train_smoke.sh

scripts/train.sh \
  --train.max_steps 2000 \
  --train.save_steps 500 \
  --train.save_epochs 0 \
  --train.num_train_epochs 1 \
  --train.use_compile false
```

正式 run 验收：

- 8 个 rank 均完成至少 2 个 optimizer steps 后再视为启动成功。
- `Ignore_Batch_Num` 始终为 0。
- loss、GradNorm 和学习率 finite。
- `global_step_500/1000/1500/2000` 同时包含 DCP 和 `hf_ckpt`。
- 结束时 torchrun 正常退出，不以日志停止更新代替退出码。

### 7.4 Open-loop 评测

```bash
export CUDA_VISIBLE_DEVICES=0
export LINGBOT_EVAL_STEP=2000
export LINGBOT_EVAL_TRAJ_IDS="0 10 20 30 43"
export LINGBOT_EVAL_MAX_INFER_TIME=3
scripts/eval_open_loop.sh
```

推理使用 `global_step_*/hf_ckpt`，不是 DCP `model/`。输出逐 action 维度曲线、MSE、MAE 和推理耗时。

### 7.5 下一次有效实验必须增加 held-out split

当前 2000-step run 使用了全部 44 episodes。下一次实验建议：

```text
train: 36-40 complete episodes
validation: 4-8 complete episodes
split unit: episode/collection condition, never frame
norm stats: train only
checkpoint selection: validation + rollout
```

需要为 train subset 建立独立 prepared receipt、contract 和 norm manifest。不能从全量统计中删除 validation 行后继续复用旧 manifest。

### 7.6 真机前检查

先做 shadow test，只记录模型动作，不下发控制：

- 相机按画面角色复用训练 mapping，不按修线后的新名字直接替换。
- 模型返回紧凑 8D action：7D 右臂绝对目标 + 1D 夹爪绝对目标。
- 原始 `[7:14]` 未训练通道保持当前值。
- 检查 joint order、单位、方向、joint limit、速度、加速度、jerk 和夹爪范围。

通过后低速闭环，初期只执行 action chunk 前 1-4 步后重新规划。最终报告任务成功率、错误抓取、掉落、碰撞、人工干预和完成时间。

## 8. 资源与存储计划

### LingBot-VLA v2

当前 2000-step run：

```text
GPU: 8 x H200
wall time: about 98 minutes under later GPU contention
checkpoint: about 92 GiB each
four checkpoints: about 368 GiB
```

启动前至少为四个 checkpoint、日志和评测预留 500 GiB。若只保留最佳 checkpoint，必须在评测完成后删除，不要边训练边删除恢复点。

### FastWAM

当前样本级 checkpoint 规模：

```text
Stage 1 lightweight checkpoint: about 10 GB
Stage 2/3 lightweight checkpoint: about 12 GB
Accelerate state: additionally includes optimizer/scheduler/RNG/sampler
```

正式训练前先通过 2-GPU resume 测试并实测每 step 吞吐、峰值显存和完整 state 大小，再决定 8-GPU checkpoint 周期。不要直接按 100k-step 上限估算完成时间。

## 9. 统一监控和停止条件

每 100-500 step 至少检查：

- 训练进程/rank 是否全部存活。
- GPU 显存、利用率和是否出现其他作业竞争。
- loss、GradNorm、学习率和 ignored batch。
- 数据吞吐、视频解码耗时和 cache hit rate。
- checkpoint 是否完整写入，是否残留临时目录。

出现以下任意情况立即停止，不继续“观察一会儿”：

- NaN/Inf 或连续异常 GradNorm。
- action mask 与 active dimensions 不一致。
- FastWAM B tier 产生非零 action loss。
- memory 读取到当前/未来窗口帧。
- LingBot 相机画面角色与训练 mapping 不一致。
- checkpoint 不可恢复或 HF shards 不完整。
- validation 指标持续恶化而 training loss 继续下降。

## 10. 推荐执行优先级

按以下顺序推进：

1. **P0：冻结 FastWAM 代码**。创建个人 fork，提交当前 memory/canonical/sampler 修改和 tests。
2. **P0：完成下载证明**。生成九库 `status` 和 `verify` 报告，补齐 LingBot-VA/DreamZero 全量数据。
3. **P0：全量数据处理**。运行九库 pipeline、视频验收、TrainingCase validation 和 train-only stats；随后把 LingBot-VA/DreamZero manifests 加入 FastWAM Stage 1/2 configs，并重建 text cache。
4. **P1：九库 smoke 回归**。Stage 1、Stage 2、Stage 3 各完成 optimizer/checkpoint/resume。
5. **P1：2-GPU 分布式验收**。确认 balanced sampler、gradient accumulation 和 resume。
6. **P2：FastWAM Stage 1 正式训练**。先评测 1k/5k/10k，不直接承诺 100k。
7. **P2：FastWAM Stage 2 Memory 正式训练**。strict 加载选定 Stage 1 checkpoint。
8. **P2：FastWAM Stage 3**。先 RoboCOIN，再目标自采数据。
9. **并行：LingBot-VLA held-out 重训**。保留 4-8 个完整 episodes，重新计算 train-only stats。
10. **最终：两模型真机对照**。使用相同任务、初始条件和安全边界比较成功率与干预率。

## 11. Definition of Done

### FastWAM 完成标准

- [ ] FastWAM 修改已提交到可访问的固定 commit。
- [ ] 九库下载和全量 manifest 有机器可读完整性报告。
- [ ] 所有 A 级 domain 有 train-only normalization stats。
- [ ] Stage 1/2/3 均完成单卡 smoke、2-GPU smoke、checkpoint 和 resume。
- [ ] Stage 2 memory 因果性、A/B loss mask 和 inference 回归通过。
- [ ] 至少一个 Stage 3 数据集有 held-out action 指标和 rollout 结果。

### LingBot-VLA v2 完成标准

- [x] 全量自采数据微调链路和 2000-step baseline 完成。
- [x] HF checkpoint 加载和 open-loop replay 完成。
- [ ] episode-level held-out 实验完成。
- [ ] shadow test 完成且动作安全检查通过。
- [ ] 低速 closed-loop rollout 有可复现成功率报告。

## 12. 关联文档

- `DATA_DOWNLOAD.md`：九库下载、恢复和完整性验证。
- `CLEANING_PIPELINE_V2.md`：清洗、分层和质量门槛。
- `ACTION_DATA_ADMISSION.md`：每库 action 进入 A 级的证据。
- `FASTWAM_TRAINING_INTEGRATION.md`：TrainingCase 到 FastWAM batch/loss。
- `THREE_STAGE_FASTWAM_TRAINING.md`：三阶段模型和 memory 细节。
- `LINGBOT_VA_DREAMZERO.md`：LingBot-VA/DreamZero 数据 schema。
- `LingBot-VLA-v2-Custom-Finetune/docs/TRAINING.md`：LingBot-VLA 训练参数。
- `LingBot-VLA-v2-Custom-Finetune/docs/EVALUATION.md`：推理和真机测试。
