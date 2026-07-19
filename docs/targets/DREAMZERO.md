# DreamZero Target

[Target 输出索引](README.md)

本文对应 [`dreamzero0/dreamzero`](https://github.com/dreamzero0/dreamzero) 的 GEAR 数据合同。

## 官方合同

当前 target profile 固定到：

```text
repository: https://github.com/dreamzero0/dreamzero
revision: ab790c198fbce33503358efbbd4187ce9a89adf3
input: LeRobot v2
default video window: 33 frames
default action horizon: 24
```

DreamZero 不使用固定 30D action。它通过 `meta/modality.json` 把 packed state/action 切成有名称的 sub-keys，再由 Hydra profile 决定相机拼接、relative action 和 normalization。

## 已提供 Profile

`configs/targets/dreamzero.yaml` 包含：

- `droid`：保留并验证官方 `modality.json`、stats 和 `oxe_droid` 注册。
- `take_wrong_item_right_arm`：15D state/action 中使用右臂 `[0:7]` 和右夹爪 `[14:15]`，三路画面角色映射沿用同一份自采数据的显式声明。

自采 profile 将 arm joint target 设为 relative-action key；gripper 保持 absolute target。

天机四路数据线标签与物理画面不一致。DreamZero 当前只使用三路，固定合同为：

| 模型顺序 | LeRobot key | 物理画面 | 2x2 位置 |
|---:|---|---|---|
| 0 | `observation.images.left_eye` | 全局/头部 | 左上 |
| 1 | `observation.images.right_eye` | 左腕 | 左下 |
| 2 | `observation.images.right_wrist` | 右腕 | 右上 |

右下保持黑屏。未使用的 `observation.images.left_wrist` 不进入这个 profile。该顺序与
DreamZero/AgiBot 预训练的 head、left hand、right hand 语义一致，不能按线缆文件名重新排序。

## Prepare

```bash
python3 scripts/prepare_dreamzero_target.py prepare \
  --source-root /data/lerobot_v2_dataset \
  --output-root /data/targets/dreamzero_dataset \
  --profile take_wrong_item_right_arm \
  --link-mode symlink \
  --verify-files
```

输出：

```text
meta/modality.json
meta/embodiment.json
meta/stats.json
meta/relative_stats_dreamzero.json
meta/dreamzero_hydra_patch.yaml
meta/dreamzero_training_profile.yaml
meta/dreamzero_target_receipt.json
data/
videos/
```

若源 Parquet 没有语言列，prepare 会从 episode tasks 生成 `annotation.task` 并重写输出 Parquet；原始数据不修改。统计包含 exact moments 和固定 seed reservoir quantiles，receipt 记录实际 quantile sample count。

## GEAR Modality

自采 profile 生成的主要 mapping：

```text
state.right_joint_position   <- observation.state[0:7]
state.right_gripper_position <- observation.state[14:15]
action.right_joint_position  <- action[0:7]
action.right_gripper_position<- action[14:15]
annotation.task              <- episode task
```

`dreamzero_hydra_patch.yaml` 同时生成：

- 33 个 video delta indices。
- 24 个 action delta indices。
- state/action/video/language modality keys。
- q99 normalization modes。
- video/state/action concat order。

## Upstream Registration Gate

自采数据使用上游已有的 `EmbodimentTag.XDOF` 和 projector mapping，只需安装数据专用
Hydra profile：

```text
meta/dreamzero_training_profile.yaml
  -> <dreamzero>/groot/vla/configs/data/dreamzero/xdof_relative.yaml
```

安装器会先检查 enum 和 projector，不修改模型代码：

```bash
python3 scripts/install_dreamzero_profile.py \
  --target-root /data/targets/dreamzero_dataset \
  --dreamzero-repo /code/dreamzero
```

训练 launcher 在启动前会再检查安装后的 YAML。

验证命令：

```bash
python3 scripts/prepare_dreamzero_target.py validate \
  --root /data/targets/dreamzero_dataset \
  --verify-files
```

## 当前限制

- 输入必须是 LeRobot v2；原始 RLDS/HDF5 视频转换应作为独立 source materialization 阶段。
- 自定义 profile 的 Hydra patch 不会自动修改外部 DreamZero checkout。
- 三相机顺序影响正迁移，不能按文件名猜测角色。
- 训练权重、DeepSpeed、LoRA 和模型 checkpoint 不属于本数据代码仓库。

## 训练入口

`scripts/run_dreamzero_training.py` 保留官方 Hydra/Trainer 路径，但允许显式指定 checkpoint，并在
`save_lora_only=true` 时跳过上游额外的 16.5B 全模型导出。`checkpoint-N/` 仍保留 LoRA、
optimizer、scheduler、RNG 和 Trainer state。命令见
[三模型统一微调](../training/THREE_MODEL_TUNING.md)。

## GT Observation Pair 推理

LoRA checkpoint 只有约 0.4 GB adapter，不包含冻结的 16.5B DiT。先把
[`dreamzero_lora_inference.patch`](../../integrations/dreamzero/dreamzero_lora_inference.patch)
应用到固定 revision；它按训练时相同顺序加载完整 base、注入 LoRA，再加载 adapter，并允许
扩大推理 KV cache。

8 卡正式推理使用四个双卡 CFG worker，每组处理两个 case：

```bash
python3 scripts/run_dreamzero_pair_inference_8gpu.py \
  --dreamzero-repo /code/dreamzero \
  --dataset-root /data/targets/take_wrong_item_dreamzero \
  --base-model-root /models/DreamZero-AgiBot \
  --checkpoint /runs/dreamzero/checkpoint-500 \
  --output-dir /runs/dreamzero/pair_inference \
  --gpus 0,1,2,3,4,5,6,7 \
  --num-cases 8 \
  --num-chunks 114 \
  --cache-window-chunks 24
```

单个 case 必须使用两卡 classifier-free guidance；编排器按 `(0,1)`、`(2,3)`、`(4,5)`、
`(6,7)` 启动四个隔离 worker，任一 worker 失败时终止其余 worker。全部完成后，它按原 episode
顺序校验八份 case receipt，并生成统一 7304 帧 reel 和 aggregate receipt。当前 H200 正式配置
使用 114 chunks，即每个 case 913 帧、约 32.6 秒；10 chunks 的 81 帧只作为最低协议门槛，
不是输出上限。默认选择可完整覆盖该 horizon 的最长八条 episode：
`7, 8, 32, 3, 2, 20, 22, 10`，其中最短 episode 仍有 920 个源帧。每个 case 的因果协议为：

- chunk 0 注入 GT 第 0 帧。
- 后续每 8 个源帧重新规划，注入截至当前时刻的最近 4 帧 GT observation。
- 当前 chunk 从不看到其目标区间内的未来 GT，也不把生成 latent 当下一次 observation。
- 每个 chunk 生成 2 个 latent，即 8 个源视频帧；114 chunks 加首帧得到严格 913 帧。
- 每次输出 24-step action horizon，Pair 指标只对齐重新规划前实际采用的前 8 steps。
- 解码每个预测块时使用同一时刻已观测 GT latent history，避免 VAE 偷换成 open-loop 预测历史。
- 总 horizon 与 cache horizon 分离。DiT KV cache 使用 `24+24+24+24+18` 五段；在 chunk
  `24, 48, 72, 96` 释放旧 cache，并用该边界时刻已经可见的四帧 GT observation 重新锚定。
  边界生成结果中的 GT anchor latent 会被剥离，只保留该 chunk 的两个预测 latent。
- GT causal VAE context 也限定为同样的 24-chunk 段，因此单次 encode/decode 最多处理 193 个
  源帧或 49 个 latent。最后 18-chunk 段有 145 帧；后处理用末帧因果 padding 48 帧以复用
  193-frame compiled graph，随后立即丢弃 padding 对应 latent，padding 不进入视频或指标。

长视频后处理不能在 Torch Compile 推理之后再次调用上游 composed CPU transform；该路径可能
等待失效的 torchvision worker pool。脚本独立复刻 XDOF 的确定性 eval 图像路径：`0.95`
中心裁剪、`176x320` antialiased bilinear resize、uint8 转换及 DreamTransform 2x2 布局，
并按 4 帧小批次执行。真实 episode 上已和上游 transform 做逐像素相等校验。rank 1 也不会用
单个 barrier 等待全部后处理；transform、每个 GT VAE segment、每个 causal decode 和最终写盘
分别同步，避免长视频后处理累计超过 NCCL 600 秒 watchdog。193 帧显存探针先验证了 8/8 case
和 24-chunk 约 130 GB/H200 的显存边界。正式 114-chunk 运行随后完成：8 个 case 均为 913 帧，
汇总 reel 为 7304 帧；32 个 case MP4 和 1 个 reel 共 36,520 帧均用 PyAV 实际逐帧解码，首尾帧
非空。八份 receipt 的 GT observation frame IDs、四次 cache reset、五段 causal VAE context 和
最后 48 帧仅用于编译形状的 padding 均通过审计。三相机 PSNR 的 case/view 总平均为
`20.7243 dB`，8D action MAE 的 case 平均为 `0.06543`。这些是当前 overfit 数据上的离线 Pair
指标，不代表真实机器人闭环成功率。

每个 episode 输出三路单视角 Pair、一路三视角总览 Pair、action NPZ 和 JSON receipt；总目录还包含
8-case reel 与 aggregate receipt。receipt 明确记录每个 chunk 注入的 GT frame IDs，便于审计
observation 是否真的来自 GT，同时记录 cache reset chunk/source frame、VAE padding、CUDA 峰值
以及每段采样耗时。
