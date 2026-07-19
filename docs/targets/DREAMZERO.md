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

官方 `droid` profile 已注册，可在 metadata 验证通过后进入训练。新的 `xdof` 或其他 embodiment 仍需把生成 patch 合并到 DreamZero 的：

```text
groot/vla/data/schema/embodiment_tags.py
groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml
groot/vla/configs/data/dreamzero/<embodiment>_relative.yaml
```

合并前 `valid=true` 只表示数据合同正确，`ready_for_training=false`。

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
