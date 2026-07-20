# RMBench Helios MemoryFastWAM

本文定义 RMBench 数据到 Helios-memory FastWAM 的训练和闭环评测合同。实际完成状态只记录在
[Validation Status](../reference/VALIDATION_STATUS.md)，运行生成的逐任务分数以
`fastwam_helios/evaluation/leaderboard_comparison.json` 为准。

## 1. 目标与边界

- 训练数据使用官方 `demo_clean`，每任务 50 条 demonstration。
- 只覆盖论文表中的 9 个任务：5 个 `M(1)` 和 4 个 `M(n)`。
- 模型是本项目恢复并修正过泄漏问题的 Helios 8/2/1 visual-memory FastWAM，不是 Mem-0。
- 每个任务独立微调并产生独立 checkpoint；不把任务间 normalization 混用。
- 闭环推理读取真实 simulator observation。生成视频或 latent 永远不回注为 observation。
- 评测使用 `unseen` instruction 和 100 个通过 expert policy 检查的 seed。

论文的 Mem-0 在 `M(n)` 任务中还包含 high-level planning 和 subtask-end classifier；当前
FastWAM 是统一 end-to-end action policy。因此结果反映的是不同架构的功能比较，不能解释为
Mem-0 的等价复现。

## 2. 固定数据快照

准备收据：

```text
/public/interns/hubin/dataset/RMBench/fastwam_helios/meta/preparation_receipt.json
```

当前 manifest SHA-256：

```text
e24fff6bf55e8f1041d55544c74d47ef8632c47d60fc5d8d04e6d2ae12dd6163
```

| Task | TMC | Episodes | Frames | Valid 81-frame windows |
|---|---:|---:|---:|---:|
| `observe_and_pickup` | M(1) | 50 | 7,523 | 3,523 |
| `rearrange_blocks` | M(1) | 50 | 20,135 | 16,135 |
| `put_back_block` | M(1) | 50 | 17,612 | 13,612 |
| `swap_blocks` | M(1) | 50 | 30,067 | 26,067 |
| `swap_T` | M(1) | 50 | 17,183 | 13,183 |
| `battery_try` | M(n) | 50 | 33,562 | 29,562 |
| `blocks_ranking_try` | M(n) | 50 | 73,789 | 69,789 |
| `cover_blocks` | M(n) | 50 | 51,127 | 47,127 |
| `press_button` | M(n) | 50 | 26,352 | 22,352 |
| **Total** | - | **450** | **277,350** | **241,350** |

准备过程只生成 manifest、normalization 和 text embedding sidecar；图像仍从官方 HDF5
直接读取，不复制成第二套视频数据。

## 3. 时间轴与图像布局

每个训练样本使用：

```text
state:   81 control points
action:  80 next-qpos targets
video:   21 frames at offsets 0, 4, ..., 80
memory:  8 long + 2 mid + 1 short observations
image:   [3, 21, 384, 320]
```

三个相机按训练和推理完全相同的方式组成一张图：

```text
head camera:        resize to 256 x 320
left wrist camera:  resize to 128 x 160
right wrist camera: resize to 128 x 160

final 384 x 320 = head on top + [left wrist | right wrist] on bottom
```

这里的 81 是 FastWAM 的训练控制窗口，不是生成或 Pair 视频的长度限制。DreamZero 的
GT/predicted Pair 使用独立时间合同。

## 4. Action 与 Proprio

RMBench Aloha-AgileX 原始向量为 14D：

```text
[left arm 6, left gripper 1, right arm 6, right gripper 1]
```

训练时映射到 FastWAM canonical 80D：

```text
source indices:    0  1  2  3  4  5  6   7  8  9 10 11 12 13
canonical slots:  14 15 16 17 18 19  6  21 22 23 24 25 26 13
```

其余 66 维固定为 0，并同时提供 valid/padding mask。Flow-matching 初始噪声、每次去噪更新和
loss 都应用相同 mask，避免无效维产生动作。state 和 next-qpos action 分任务做 z-score；
评测时用相同 domain stats 反归一化，再从 canonical 80D 取回 14D qpos。

## 5. Memory 合同

训练样本的 memory frame 必须严格早于当前 81-frame target window。episode 开头历史不足时：

- 用当前首帧左填充，以保持 8/2/1 tensor shape。
- 对填充位置设置 `memory_mask=false`。
- memory patch token 在进入 attention 前乘 mask。
- target frame 不允许作为有效 memory。

闭环评测时 server 保存最近 11 个真实 simulator observation。每次 replan 将
`history + current observation` 交给 memory selector；selector 排除最后的 current frame，
因为 current 已作为 FastWAM `input_image/x0`。每个 action chunk 只执行前 8 个 qpos；在
第 4 个 action 后注入一次真实 simulator observation，第 8 个 action 后由下一次 replan 读取
新的真实 observation。该 cadence 与训练视频的 4-control-step stride 一致，不使用模型生成图像
作为 observation。

## 6. 训练配方

初始化 checkpoint 是天机 44 条自采数据完成的 Helios-memory FastWAM：

```text
.../fastwam_tianji_dataset_overfit_cont_1250_to_2500/
checkpoints/weights/step_001250.pt
```

文件名中的 1,250 是 continuation-local step；它接续前一阶段 1,250 steps，累计为 2,500
self-data steps。RMBench 每个任务的配置为：

| Field | Value |
|---|---:|
| batch size | 1 per task/GPU |
| task fine-tune steps | 2,500 |
| optimizer | AdamW |
| learning rate | `1e-5` |
| schedule | cosine |
| precision | BF16 |
| gradient checkpointing | enabled |
| loss | video 1.0 + action 1.0 |
| save interval | 500 steps |
| memory tokens | 168 |

8 张 GPU 同时运行前 8 个任务，首个空闲 GPU 再运行第 9 个任务。每个保存点同时写：

- 轻量 policy checkpoint：`checkpoints/weights/step_XXXXXX.pt`。
- 完整 Accelerator state：model、optimizer、scheduler、random state 和 `trainer_state.json`。

launcher 只从字段完整且 `global_step` 匹配的 state 恢复。最终 checkpoint 存在但 receipt
不匹配时会拒绝静默覆盖。

当前 9/9 个任务均已完成 2,500 steps。训练完整性检查逐项读取了 5 个保存点：每任务的
`500/1000/1500/2000/2500` 轻量 checkpoint 均为有效 torch archive，同时对应的 model、
optimizer、scheduler、random-state 和 `trainer_state.json` 均存在且 global step 一致，
即 45/45 个轻量 checkpoint 和 45/45 份完整训练状态通过。

```bash
cd /path/to/FastWAM
python scripts/rmbench/launch_rmbench_helios_training.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --steps 2500
```

## 7. 闭环评测

FastWAM 和 RMBench 分属不同 Python 环境，通过 localhost socket 隔离。每个任务有独立：

- FastWAM model server。
- RMBench simulator/client。
- CUDA device 和 TCP port。
- XDG/Vulkan runtime directory。
- progress JSON、日志、视频和 result receipt。

本节点 H200 没有向 SAPIEN 暴露 Vulkan graphics queue，因此 simulator 使用 Mesa lavapipe
运行 SAPIEN `default` raster camera shader，CuRobo 和 FastWAM 仍使用 CUDA。CPU `rt`
路径即使降到 1 spp/depth 1，仍在长 rollout 中触发 Mesa native camera segmentation fault，
不适合 9 x 100 正式评测。官方环境使用 GPU ray-tracing renderer，两者不是
pixel-equivalent。renderer pipeline、shader、适用参数和 Vulkan device 都写入 receipt，并由
比较脚本动态读取。

Mesa CPU `rt` 的 `take_picture/get_picture` 会在长 rollout 中随机触发 native segmentation
fault；显式选择 `rt` 时脚本因此默认用主机级 `fcntl` 锁保护完整 observation 事务。正式
`default` raster 协议不使用跨进程渲染锁，8 个 simulator/model pair 可同时推进。每个任务使用
独立 XDG cache/tmp/Mesa shader cache；torch、BLAS 和
OpenMP 均限制为 1 thread，lavapipe 固定为 `LP_NUM_THREADS=1`，避免驱动内部并发和 9 个进程
过度分配 CPU 线程。FastWAM 输入不消费相机内外参，因此评测跳过每 step 的 camera-matrix
native 读取；三路 RGB、qpos 和相机布局不变。

```bash
RMBENCH_RENDER_SHADER=default \
RMBENCH_LP_NUM_THREADS=1 \
RMBENCH_SKIP_CAMERA_CONFIG=1 \
RMBENCH_OBSERVATION_STRIDE=4 \
python scripts/rmbench/launch_rmbench_helios_evaluation.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 2 \
  --max-attempts 20 \
  --launch-stagger-seconds 20 \
  --step 2500
```

9 个任务按 breadth-first 分配：先各占一张 GPU，第九个任务才使用 GPU 0 的第二个 slot。
每个 rollout 完成后原子写入下一个 seed、完成数和 success 数；中断后从下一条 rollout 恢复。
launcher 对 native renderer 退出执行有界重试，每次重试复用原子 progress，而不是重新统计已完成
rollout。progress identity 同时固定 checkpoint、renderer pipeline 和 observation stride；raster
协议将 spp、path depth、denoiser 记为 null，避免伪造不适用参数。任一结果协议变化都会拒绝读取
旧进度。显式 RT 锁只改变任务间调度顺序，不改变单个 observation 的相机、分辨率或像素后处理。

正式 result 只有同时满足以下条件才进入排行榜：

1. receipt 指向 exact `step_002500.pt`，且 checkpoint 是完整 torch archive。
2. progress 为 `completed` 且 `evaluated_rollouts=100`。
3. progress identity 的 `target_rollouts=100` 且 checkpoint 路径一致。
4. 存在 100 个不重复的 expert-feasible seed。
5. progress success 计数与官方 `_result.txt` 数值一致。

## 8. 对比口径

论文 Mem-0 使用 30K iterations、global batch 448、8 x A800。当前公开权重使用：

- M(1)：五任务联合 checkpoint，50K steps、batch 56。
- M(n)：四个 task-specific checkpoint，每个 30K steps。

当前 FastWAM 使用 Tianji 2.5K-step 初始化，再对每个 RMBench task 微调 2.5K steps。因此比较
既不匹配训练预算，也不匹配 M(1) 的联合训练组织；报告必须同时保留论文 Mem-0 和当前公开
Mem-0 两列，并明确 renderer 差异。

```bash
python scripts/rmbench/summarize_rmbench_results.py
```

输出：

```text
/public/interns/hubin/dataset/RMBench/fastwam_helios/evaluation/
  leaderboard_comparison.json
  leaderboard_comparison.md
```

参考来源：RMBench [project](https://rmbench.github.io/)、
[paper](https://arxiv.org/html/2603.01229)、
[official code](https://github.com/robotwin-Platform/rmbench)、
[current M(1) checkpoint](https://huggingface.co/qiuly/Mem-0-m1mix-RMBench) 和
[current M(n) checkpoints](https://huggingface.co/qiuly/Mem-0-mn-RMBench)。
