# FastWAM 天机机械臂单轨迹过拟合

[文档索引](../README.md)

更新日期：2026-07-19

本文定义 `take_wrong_item_right_arm` 上的 FastWAM 过拟合实验。目标不是证明模型泛化，
而是先回答一个更基础的问题：固定同一个历史、首帧、语言和 proprio 后，Memory FastWAM
能否记住该窗口的未来视频与动作，并稳定输出“想象未来 vs GT”的可视化结果。

> 本文保留的是单窗口链路诊断及 2026-07-19 已完成结果。当前“保留原始鱼眼并使用全部 44 条轨迹”
> 的正式实验见 [FastWAM 天机全采集数据原始鱼眼域过拟合](FASTWAM_TIANJI_DATASET_OVERFIT.md)。
> 本文历史 checkpoint 使用原始鱼眼画面，不能与新实验的虚拟针孔指标直接比较。

## 1. 完成定义

一次有效实验必须同时满足：

1. 训练和评测使用同一个冻结窗口，不允许每次评测随机换样本。
2. step 0、训练中间点和最终点使用相同 inference seed、相同 diffusion steps。
3. 推理真实消费 8/2/1 memory，而不是退化成仅首帧推理。
4. 视频同时报告 rollout vs GT、rollout vs VAE reconstruction、VAE vs GT。
5. 动作只在 8 个有效 canonical slots 上评测，并输出逐 slot 误差。
6. demo 明确标注 `IMAGINATION`、`VAE RECONSTRUCTION`、`GROUND TRUTH`。
7. checkpoint、日志、视频和 action artifact 均在 `work/`，不进入代码仓库。

只看到 training loss 下降不能判定成功。diffusion loss 随 timestep 和 noise 变化，若不固定评测
随机数，它甚至不能用于跨 step 比较。

## 2. 冻结样本

当前 overfit phase 固定：

| 字段 | 值 |
|---|---|
| source dataset | `take_wrong_item_right_arm` |
| FastWAM case id | `f0760af5c9dfd420f5af1854` |
| source episode | `take_wrong_item_right_arm:000000:xdof_right_joint_15d` |
| dataset sample index | `0`，在 `sample_offset=1` 后 |
| canonical window start | `40` |
| video | `[3,21,384,320]` |
| action / proprio | `[80,80]` / `[80,80]` |
| valid action slots | `[13,21,22,23,24,25,26,27]` |
| memory valid count | long 8、mid 2、short 1 |

不选原始 sample 0，因为它从 episode 起点开始，11 个历史 memory token 都是无效 padding。
start 40 已有完整历史，而且有效 action 的平均绝对 z-score 为约 `0.490`、平均时序标准差为约
`0.326`，不是静止窗口。

## 3. 相机与画面布局

采集时相机数据线接反，必须按已审计的画面内容绑定：

| 原始 key | 实际物理视角 | FastWAM role |
|---|---|---|
| `observation.images.left_eye` | 头部/顶部全局相机 | `global_primary` |
| `observation.images.right_eye` | 左腕相机 | `left_wrist` |
| `observation.images.right_wrist` | 右腕相机 | `right_wrist` |

最终 384x320 composite 的上方 256x320 是全局视角，下方两个 128x160 区域依次为左腕和
右腕。训练和 demo 都使用这个布局，因此不会受错误 key 名影响。真实部署时也必须复用同一
物理绑定，不能根据字符串名称临时改回去。

## 4. 模型条件与限制

当前 `fastwam_memory_canonical` 的视频条件是：

```text
text + initial composite frame + initial proprio + leak-free 8/2/1 visual memory
```

action expert 从同一上下文预测 80 步 action。当前配置
`video_dit_config.action_conditioned=false`，因此传入 GT action 不会条件化视频生成。
demo 应称为“FastWAM imagined future vs GT”，不能称为“GT-action-conditioned rollout”。

若后续需要反事实控制，例如给同一首帧输入两条不同 action 并生成两个不同未来，需要单独
启用和训练 action-conditioned video expert，或修改 MoT 的 video-to-action attention。这不是
本次单轨迹过拟合的隐式功能。

## 5. 训练设置

`configs/tuning/take_wrong_item.example.yaml` 的 `fastwam.phases.overfit` 使用：

| 配置 | 值 | 原因 |
|---|---:|---|
| train samples | 1 | 严格诊断记忆能力 |
| batch / grad accumulation | 1 / 1 | 每个 optimizer step 都是同一窗口 |
| backbone learning rate | `2e-5` | 比常规 Stage 3 更快地记忆单窗口 |
| memory patcher learning rate | `1e-4` | 允许三尺度 memory adapter 快速适配 |
| video / action loss | `1.0 / 1.0` | 共享 MoT 上保持两类梯度均衡 |
| weight decay | 0 | 过拟合诊断不施加泛化正则 |
| scheduler | constant | 便于解释 step 数 |
| max steps | 命令行指定，首轮 300 | 先观察 0/50/100/150/200/250/300 |
| eval steps | 10 | 固定成本的 diffusion rollout |
| eval seed | 42 | 所有 step 可比 |
| save / eval interval | 50 / 50 | checkpoint 与 demo 一一对应 |
| full trainer state | false | 避免每次额外保存约 36GB optimizer state |

初始化使用 Stage 2 memory checkpoint，不从已经看过该目标的 Stage 3 checkpoint 开始。这样
step 0 才是有效基线。

## 6. 启动

新服务器先把代码仓库携带的 integration 应用到固定 FastWAM 基线：

```bash
scripts/apply_fastwam_integration.sh --check /path/to/FastWAM
scripts/apply_fastwam_integration.sh --apply /path/to/FastWAM
```

补丁范围、上游 SHA 和干净 checkout 要求见
[`integrations/fastwam/README.md`](../../integrations/fastwam/README.md)。统一 launcher 的 overfit
preflight 还会检查 `infer_joint`、`infer` 和 trainer 是否具备 memory-aware 固定评测能力，避免
在旧版 FastWAM 上启动数小时后才失败。

先验证命令和路径：

```bash
python3 scripts/tune_models.py dry-run \
  --config work/tuning/take_wrong_item.local.yaml \
  --model fastwam \
  --phase overfit \
  --output-dir work/tuning/runs/fastwam_tianji_overfit \
  --steps 300 \
  --gpus 0
```

确认目标 GPU 至少有约 70GB 可用显存后启动：

```bash
python3 scripts/tune_models.py run \
  --config work/tuning/take_wrong_item.local.yaml \
  --model fastwam \
  --phase overfit \
  --output-dir work/tuning/runs/fastwam_tianji_overfit \
  --steps 300 \
  --gpus 0
```

当前 H200 上单卡训练实测占用约 63GB。不能在只剩 30GB 左右的卡上并行启动，也不能终止
其他人的任务来抢卡。

## 7. 输出

```text
work/tuning/runs/fastwam_tianji_overfit/
  _wm_tuning/                    launcher 收据和完整日志
  checkpoints/weights/           step 50、100...轻量模型权重
  eval/
    step_000000_rank_000.mp4      未训练基线三联视频
    step_000000_rank_000.json     固定评测指标和 conditioning 声明
    step_000000_rank_000_actions.npz
    step_000050_rank_000.mp4
    ...
```

JSON 的缩写含义：

| 字段 | 含义 |
|---|---|
| `psnr_rg` / `ssim_rg` | rollout vs ground truth，主指标 |
| `psnr_rd` / `ssim_rd` | rollout vs VAE decode，排除 VAE 上限影响 |
| `psnr_dg` / `ssim_dg` | VAE decode vs ground truth，重建上限 |
| `action_l1` | 8 个有效 slot 的反归一化平均绝对误差 |
| `action_l1_by_slot` | 逐 canonical slot 的反归一化 MAE |
| `memory_valid_ratio` | 实际有效 memory token 比例，当前应为 1.0 |

`canonical_denormalized_mixed_units` 表示逐 slot 已回到源域单位，但全局 `action_l1` 混合了
关节和 gripper 单位。模型选择时必须同时查看逐 slot 值，不能只解读一个全局数。

ActionDiT 还有一个必须保持的推理契约：`action_dim_is_pad=true` 的 72 个无效 canonical
维度在训练时为零，因此推理也必须在随机初始化和每个 flow-matching 去噪步后重新置零。
若让全部 80 维都带随机噪声，动作线性嵌入会把未训练维度污染到 8 个有效维度。trainer
必须把样本的 `action_dim_is_pad` 传给 `model.infer()`；不能只在计算 metric 时使用该 mask。

## 8. 自动汇总与门槛

训练结束后执行：

```bash
python3 scripts/summarize_fastwam_overfit.py \
  --run-dir work/tuning/runs/fastwam_tianji_overfit
```

若训练使用轻量 checkpoint 续跑，global step 会从 0 重新计数。用原始 run 的 step-0 作为
统一基线，并显式给续跑 step 加 offset：

```bash
python3 scripts/summarize_fastwam_overfit.py \
  --run-dir work/tuning/runs/fastwam_tianji_overfit_continue \
  --baseline-run-dir work/tuning/runs/fastwam_tianji_overfit \
  --step-offset 300 \
  --output-dir work/tuning/runs/fastwam_tianji_overfit_verified
```

它生成 `overfit_report.json` 和 `OVERFIT_REPORT.md`。默认候选 checkpoint 必须同时满足：

- 和 step 0 是同一冻结窗口。
- `memory_valid_ratio >= 0.999`。
- rollout vs GT PSNR 相对 step 0 至少提高 3 dB。
- rollout vs GT SSIM 相对 step 0 至少提高 0.05。
- action L1 不高于 step 0 的 50%。

这些是过拟合诊断门槛，不是最终部署指标。若 300 step 未通过，先看三条曲线分别判断：

1. video 与 action 都不下降：检查 checkpoint 是否正确加载、参数是否真的 trainable。
2. action 下降但 video 不动：提高 video loss 权重或只训视频分支做隔离实验。
3. video 下降但 action 不动：检查有效 action mask、normalization 和 action expert 初始化。
4. loss 下降但 rollout 不改善：增加 inference steps，并检查训练 scheduler 与 inference scheduler。
5. rollout 接近 VAE 但仍模糊：这是 VAE reconstruction ceiling，不应继续归因于 DiT。

## 9. 真实运行结果

2026-07-19 在单张 H200 上完成了 300 optimizer steps，launcher receipt 状态为
`succeeded`，总耗时 `405.5 s`。每 50 步保存一个约 12GB 的轻量 checkpoint；未保存约
36GB 的 optimizer state。

| 累计 step | val loss | rollout vs GT PSNR | rollout vs GT SSIM |
|---:|---:|---:|---:|
| 0 | 0.6428 | 10.5547 | 0.1762 |
| 50 | 0.0301 | 17.9306 | 0.7459 |
| 100 | 0.0203 | 19.6026 | 0.7931 |
| 150 | 0.0188 | 23.0107 | 0.8591 |
| 200 | 0.0111 | 23.6236 | 0.8744 |
| 250 | 0.0100 | 24.5935 | 0.8953 |
| 300 | 0.0068 | **25.4151** | **0.9072** |

随后从 step 300 轻量权重续训 600 步，续训耗时 `748.0 s`。累计 step 900 的固定评测为
`val_loss=0.0038`、PSNR `26.8693`、SSIM `0.9313`，已经接近 VAE vs GT 的
`27.5665 / 0.9371` 上限。

修复 action padding 后，使用完全相同的 seed、10 个 inference steps 和冻结窗口重新评测：

| 指标 | Stage-2 / step 0 | overfit / step 300 | 变化 |
|---|---:|---:|---:|
| action L1 | 0.30233 | **0.02337** | 仅为基线的 **7.73%** |
| action MSE | 0.17917 | **0.00104** | 降低 99.42% |
| memory valid ratio | 1.0 | 1.0 | 全程有效 |
| VAE vs GT PSNR / SSIM | 27.5665 / 0.9371 | 27.5665 / 0.9371 | 固定重建上限 |

step 300 是首个完整验证的最小通过点。累计 step 900 的修复后 action L1 / MSE 进一步为
`0.01941 / 0.000969`，即基线 L1 的 `6.42%`。最终严格报告为 `passed`，位于
`work/tuning/runs/fastwam_tianji_overfit_verified/OVERFIT_REPORT.md`。最终选中权重是
`work/tuning/runs/fastwam_tianji_overfit_continue/checkpoints/weights/step_000600.pt`
（在原 step 300 上续训，累计 step 900）；修复后 demo 位于
`work/tuning/runs/fastwam_tianji_eval_masked_total900/eval/step_000000_rank_000.mp4`。

做过一次 `lambda_action=4` 消融。step 150 的 action L1 为 0.1924，与 1:1 的 0.1913
持平，但视频 PSNR 只有 19.80 dB，低于 1:1 的 23.01 dB。共享 MoT 配合
`max_grad_norm=1.0` 时，放大 action loss 会占用梯度裁剪预算，因此默认保持 1:1。该消融
不能替代 action padding 修复。

## 10. 过拟合后的下一步

单窗口通过后按顺序扩大，不应直接跳到全量：

1. 同一 episode 的 4 个窗口，验证模型不是只记首帧纹理。
2. 4 个 train episodes，保留 1 个 held-out episode。
3. 全部训练 split，按 episode 做 validation。
4. 在真实天机机械臂上先离线回放，再做低速、限幅、可急停的闭环推理。

单轨迹 overfit 只证明训练、memory、推理和可视化链路一致，不证明策略安全或任务成功率。
