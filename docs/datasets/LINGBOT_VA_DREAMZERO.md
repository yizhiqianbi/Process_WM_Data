# LingBot-VA and DreamZero to FastWAM

[文档索引](../README.md)

本文说明 `lingbot_va` 和 `dreamzero` 两个适配器的输入合同、相机布局、
action/state 映射、清洗门槛、下载方式和已完成验证。这里处理的是数据，不包含
LingBot-VA、DreamZero 或其他模型权重。

这里的 LingBot-VA 是数据源，不是 LingBot-VLA v2 模型。整体通过/待验状态统一见
[Validation Status](../reference/VALIDATION_STATUS.md)。

## 1. 实现范围

当前代码支持三个真实上游 schema：

| FastWAM dataset | 上游子集 | 原始格式 | FastWAM action 状态 |
|---|---|---|---|
| `lingbot_va` | RoboTwin cleaned/augmented | 28 个 LeRobot v2.1 repo | 已验证双臂 EEF/gripper 映射 |
| `lingbot_va` | LIBERO-Long | 单个 LeRobot v2.1 repo | 已验证单臂 delta EEF/gripper 映射 |
| `dreamzero` | DreamZero-DROID | LeRobot v2.0 + GEAR metadata | 已验证 Panda joint/gripper slice |

未知 LeRobot schema 仍可扫描视频、任务和 episode，但不会猜 action。它会保留
unverified mapping，并在 clean 后降级为 `video_only`。这条规则用于防止相同维度、
不同控制语义的数据被错误混合。

## 2. 固定的上游版本

`configs/download_manifest.lock.json` 当前固定到：

| Repository | Commit SHA | 当前 HF 访问状态 |
|---|---|---|
| `robbyant/robotwin-clean-and-aug-lerobot` | `595d4acc7e026aa57afba218f4170c6d79e39103` | public |
| `robbyant/libero-long-lerobot` | `8c0313b1c7cd9fa3798798479cbf59b11af8979d` | public |
| `GEAR-Dreams/DreamZero-DROID-Data` | `2abc197ca7f14f53a6bf464bf80018ce998f18cc` | public |

官方实现和数据说明：

- LingBot-VA: <https://github.com/Robbyant/lingbot-va>
- LingBot-VA RoboTwin data: <https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot>
- LingBot-VA LIBERO data: <https://huggingface.co/datasets/robbyant/libero-long-lerobot>
- DreamZero: <https://github.com/dreamzero0/dreamzero>
- DreamZero-DROID data: <https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data>
- DreamZero conversion contract: <https://github.com/dreamzero0/dreamzero/blob/main/docs/DATASET_TO_GEAR_AND_TRAIN.md>

更新 lock 是显式操作。生产训练不应在不知情的情况下从固定 SHA 切换到上游
`main`。

## 3. 统一 FastWAM 输出

两个适配器最终都进入同一合同：

- 20 Hz control timeline。
- 81 个 state 点，覆盖 4 秒。
- 80 个 action 点。
- 视频 offset 为 `0, 4, ..., 80`，共 21 帧。
- state/action 均为 80D canonical vector。
- 每一维必须同时读取 `state_dim_valid_mask` 或 `action_dim_valid_mask`。
- 视频按 `global_primary`、`global_secondary`、`left_wrist`、`right_wrist`、
  `auxiliary` 五个固定槽位组织。
- source 不会被复制进代码库；canonical Parquet 是小型数值 sidecar，视频保持引用。

上游模型参数不会覆盖这个合同：

- LingBot-VA 的模型 action width 是 30D，不代表源数据都是 30D。
- DreamZero 默认 action horizon 是 24、视频窗口是 33 帧，不代表 FastWAM case
  应改成 24/33。
- 这些上游参数记录在 `TrainingCaseV1.provenance.source_profile`，用于复现实验，
  不用于改变 FastWAM 81/80/21 输入。

## 4. LingBot-VA RoboTwin

### 4.1 数据规模和目录

当前固定 revision 包含 28 个 LeRobot metadata root：

```text
LingBot_VA/
└── robotwin-clean-and-aug-lerobot/
    └── lerobot_robotwin_eef_aug_500/
        ├── <task-a>/
        │   ├── meta/
        │   ├── data/chunk-000/
        │   └── videos/chunk-000/
        └── <task-b>/...
```

metadata 汇总为 14,000 episodes、2,827,760 source frames、50 Hz。适配器递归发现
`meta/info.json`，不会把最外层 HF repository 误认为单个 LeRobot episode repo。

### 4.2 相机

| Source key | FastWAM role | Source resolution/FPS |
|---|---|---|
| `observation.images.cam_high` | `global_primary` | 640x480, 50 Hz |
| `observation.images.cam_left_wrist` | `left_wrist` | 640x480, 50 Hz |
| `observation.images.cam_right_wrist` | `right_wrist` | 640x480, 50 Hz |

角色来自 embodiment profile，不依赖通用字符串排序。缺一台相机时，对应 slot mask
为 false，不会把另一台相机静默挪到错误的 wrist。

### 4.3 原始 state/action

`observation.state` 和 `action` 都是 16D：

```text
[left_xyz(3), left_quaternion_xyzw(4), left_gripper(1),
 right_xyz(3), right_quaternion_xyzw(4), right_gripper(1)]
```

官方 LingBot-VA RoboTwin loader 对左右 7D pose 调用 `get_relative_pose`，再按如下
顺序填入模型 30D 通道：

```text
[0..6, 28, 7..13, 29]
```

因此 30D 是模型内部统一 layout，不是 Parquet 原始 action width。

### 4.4 FastWAM 映射

FastWAM canonical EEF 使用 `xyz + rotation_vector`，不是 quaternion。预处理器在
`ParquetSourceReader` 内完成 quaternion normalization 和 quaternion-to-rotation-vector
转换：

| Source | Canonical slots | Valid count |
|---|---:|---:|
| left EEF | `0..5` | 6 |
| left gripper | `6` | 1 |
| right EEF | `7..12` | 6 |
| right gripper | `13` | 1 |

总计 14 个 state slots 和 14 个 action slots。原始 16D 列仍作为 monitor-only signal
参与 finite/quaternion 审计；训练只读取派生列，避免把 quaternion 四维直接写入
rotation-vector 三维。

canonical 中保存 source absolute EEF target。官方 LingBot 的“相对 episode/segment
首帧”变换属于模型 loader policy，不在共享 canonical sidecar 中硬编码。这样做有两个
原因：

1. FastWAM window 可以从 episode 中任意合法 start 开始，相对参考帧应由具体 loader
   定义，不能固定成 episode frame 0。
2. absolute target 允许 action/state alignment 和来源审计，后续仍可无损派生相对 action。

### 4.5 Action segmentation

LingBot 数据在 `meta/episodes.jsonl` 中增加：

```json
{
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 465,
      "action_text": "..."
    }
  ]
}
```

该字段原样进入：

```text
TrainingCaseV1.provenance.source_episode_metadata.action_config
```

当前官方 RoboTwin/LIBERO 样本通常使用覆盖整段 episode 的单 segment。若将来接入
多 segment 自定义数据，生成窗口前应增加 segment-boundary filter，避免一个 4 秒窗口
跨两个动作文本。当前代码保留了完成该过滤所需的 start/end/text，但没有擅自重写
episode task prompt。

## 5. LingBot-VA LIBERO-Long

### 5.1 数据和相机

当前 revision 报告 500 episodes、138,090 frames、60 Hz。

| Source key | FastWAM role | Shape |
|---|---|---|
| `observation.images.agentview_rgb` | `global_primary` | 128x128 |
| `observation.images.eye_in_hand_rgb` | `left_wrist` | 128x128 |

LIBERO 是单臂 Franka，`eye_in_hand_rgb` 明确放入 left/single-arm wrist slot。

### 5.2 state/action

```text
observation.state: 8D
  [x, y, z, roll, pitch, yaw, gripper_finger_1, gripper_finger_2]

action: 7D
  [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper_command]
```

处理规则：

- state 的 `xyz+rpy` 转为 `xyz+rotation_vector`。
- action 的增量 `xyz+rpy` 转为增量 `xyz+rotation_vector`。
- 第一个 gripper feedback 作为 canonical gripper state。
- action 第 6 维作为 gripper command。
- 第二个 finger feedback 保留在原始 monitor signal，不重复映射 canonical slot。

Canonical slots 为 `0..6`。EEF delta 与 state derivative 做 lagged Pearson alignment；
gripper command 和 finger feedback 量纲不同，不用于相关性门控，但两者仍被 finite、
范围和离散跳变规则检查。

## 6. DreamZero-DROID

### 6.1 数据规模和过滤来源

当前 `meta/info.json` 报告：

- 57,774 episodes。
- 14,748,517 frames。
- 15 Hz。
- LeRobot codebase version `v2.0`。

官方 DreamZero 说明该版本来自 DROID 1.0.1，并已经移除 idle frame、过滤无语言
episode、只保留 success。FastWAM 不重复假设这些过滤一定正确，仍执行自身的时序、
信号、视频和语言审计。

### 6.2 三相机布局

| Source key | FastWAM role | Source shape/FPS |
|---|---|---|
| `observation.images.exterior_image_1_left` | `global_primary` | 320x180, 15 Hz |
| `observation.images.exterior_image_2_left` | `global_secondary` | 320x180, 15 Hz |
| `observation.images.wrist_image_left` | `left_wrist` | 320x180, 15 Hz |

DreamZero metadata 使用 `video_info`，而部分 LeRobot v2.1 数据使用 `info`。通用 reader
同时支持这两个字段，并通过 `names=[height,width,channel]` 验证 shape 顺序。

### 6.3 packed state/action

`observation.state` 为 14D，`action` 为 28D。不能根据维度猜 slice，适配器只信任
官方 `meta/modality.json`：

```text
state:
  cartesian_position  [0:6]
  gripper_position    [6:7]
  joint_position      [7:14]

action:
  cartesian_position  [0:6]
  cartesian_velocity  [6:12]
  gripper_position    [12:13]
  gripper_velocity    [13:14]
  joint_position      [14:21]
  joint_velocity      [21:28]
```

DreamZero 官方训练配置实际选择：

```text
state.joint_position + state.gripper_position
action.joint_position + action.gripper_position
```

并对 `joint_position` 启用 relative action；gripper 不做 relative transform。

### 6.4 FastWAM 映射

适配器先派生四个独立列：

```text
fastwam.state.left_joint_position
fastwam.state.left_gripper
fastwam.action.left_joint_target
fastwam.action.left_gripper_target
```

只对这些列生成训练 hard interval。packed 原始列的 Cartesian/velocity 字段仍被监控，
但不会因为未使用维度的跳变而错误屏蔽 joint window。

| Source semantic | Canonical slots |
|---|---:|
| single-arm/Panda gripper | `6` |
| Panda joint 1..7 | `14..20` |

总计 8 个 state slots 和 8 个 action slots。这里把单臂统一放在 FastWAM left block，
这只是 canonical 命名约定，不表示 DROID 必须是物理左臂。

canonical 保存 absolute joint/gripper target。DreamZero 的
`action - reference_state` 仍是训练 loader policy，并记录为：

```text
provenance.source_profile.dreamzero_relative_action_keys = ["joint_position"]
```

## 7. 清洗和 admission

两个适配器共用以下门槛：

1. `meta/info.json` 和 `meta/episodes.jsonl` 可读。
2. Parquet row count 与 episode length 一致。
3. timestamp 严格递增，frame index 连续。
4. 训练相关派生列全部 finite、宽度固定。
5. quaternion norm/sign continuity、angle unwrap 和 abrupt step 审计通过。
6. state/action 存在可解释 lag alignment。
7. canonical mapping 带官方 provenance 且 `verified=true`。
8. source duration 足够生成一个 4 秒窗口。
9. 视频至少存在一个有效语义相机；深度 decode 可在第二轮启用。

针对异构控制的两项规则：

- gripper 是 zero-order/discrete signal。开合边沿保留为 soft event，不再作为连续关节的
  extreme hard jump。
- EEF pose 使用 `0.01` 每 source frame 的基础 abrupt floor，hard gate 仍由
  `extreme_abrupt_multiplier=10` 控制。普通高频运动不会因长 stationary 段导致 MAD
  过小而整段被拒绝，真正超过 0.1 的单帧位姿跳变仍会成为 hard interval。

最终 admission：

- `joint_video_action`: state/action mapping、信号和 alignment 全部验证。
- `video_only`: 视频可用，但 action schema 未知或 action audit 未通过。
- `reject`: episode 结构、时序、长度或全部视频失败。

## 8. 下载

只检查这三个 repository 的访问：

```bash
python3 scripts/download_datasets.py access \
  --datasets lingbot_va dreamzero \
  --token-file /secure/path/hf_token.txt
```

预览目的路径：

```bash
python3 scripts/download_datasets.py download \
  --datasets lingbot_va dreamzero \
  --data-root "$FASTWAM_DATA_ROOT" \
  --dry-run
```

并行、可恢复下载：

```bash
python3 scripts/download_datasets.py download \
  --datasets lingbot_va dreamzero \
  --data-root "$FASTWAM_DATA_ROOT" \
  --token-file /secure/path/hf_token.txt \
  --repo-jobs 3 \
  --file-workers 4 \
  --attempts 5
```

当前 lock 的 HF 文件 metadata 约为 RoboTwin 208 GiB、LIBERO 4.4 GiB、DreamZero
33 GiB。上游可能更新展示值；生产容量规划应以固定 revision 的 `verify` 报告和目标
文件系统实际占用为准。

## 9. 预处理命令

先跑 metadata + 一个 episode：

```bash
python3 scripts/preprocess_lingbot_va.py pipeline \
  --max-episodes 1 \
  --verify-files

python3 scripts/preprocess_dreamzero.py pipeline \
  --max-episodes 1 \
  --verify-files
```

统一入口：

```bash
python3 scripts/run_pipeline.py \
  --datasets lingbot_va dreamzero \
  --output-root work/stage_pipeline \
  --num-frames 81 \
  --target-fps 20 \
  --workers 2 \
  --verify-files
```

第一轮建议不加 `--decode-videos`。完成 manifest 和 action admission 后，再用
`--check-videos --decode-videos` 对 A/B candidate 做稀疏视觉和完整容器解码。

生成 normalization stats：

```bash
python3 scripts/build_fastwam_normalization_stats.py \
  --pipeline-root work/stage_pipeline \
  --datasets lingbot_va dreamzero \
  --data-root . \
  --output work/stage_pipeline/normalization_stats_external.json
```

stats 只读取 train split、A tier、`joint_video_action` case，并按 embodiment + active
slots 分 domain，不会把 RoboTwin EEF、LIBERO delta EEF 和 DROID joint target 混成
一个统计分布。

## 10. TrainingCase 检查

预期 active slots：

| Family | State slots | Action slots | Required camera roles |
|---|---|---|---|
| `lingbot_va_robotwin` | `0..13` | `0..13` | global + left/right wrist |
| `lingbot_va_libero_long` | `0..6` | `0..6` | global + one wrist |
| `dreamzero_droid` | `6,14..20` | `6,14..20` | two global + one wrist |

检查一个 case：

```bash
jq '{dataset,
     mode:.training.mode,
     timeline,
     state_slots:.inputs.state_valid_slots,
     action_slots:.inputs.action_valid_slots,
     cameras:[.inputs.camera_slots[] | select(.valid) | [.role,.source_key]],
     source_profile:.provenance.source_profile}' \
  work/stage_pipeline/dreamzero/cases/example_case.json
```

任何 `joint_video_action` case 都必须同时满足：

```text
timeline.state_steps  == 81
timeline.action_steps == 80
timeline.video_steps  == 21
training.loss_mask.action == true
len(inputs.state_slot_mask) == 80
len(inputs.action_slot_mask) == 80
```

## 11. 已完成验证

自动测试 `tests/test_external_datasets.py` 使用与官方 metadata 同形的 fixtures，分别跑完：

```text
scan -> clean -> materialize -> windows -> cases
```

三个分支都必须产出 `joint_video_action` 81/80/21 case。额外真实 HF 小样本验证结果：

| Sample | Clean admission | Joint windows | Active action slots |
|---|---|---:|---|
| RoboTwin `blocks_ranking_rgb`, episode 0 | joint | 3 | 14 |
| LIBERO-Long, episode 0 | joint | 1 | 7 |
| DreamZero-DROID, episode 0 | joint | 6 | 8 |

真实样本验证使用固定 SHA 下载到临时目录，不进入 Git，不进入导出的纯代码包。

运行回归：

```bash
python3 -m unittest tests.test_external_datasets -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q fastwam_preprocess scripts tests
```

## 12. 仍需完成的生产任务

当前已经证明 schema 和样本级训练路径可用，不等于两个全量数据集已经完成生产训练。
正式 Stage 1/2 前还需要：

1. 完成三个 locked repository 的全量下载与远端 size verification。
2. 对全量 episode 跑 `--verify-files`，再对 A/B candidate 做视频 decode。
3. 统计每个 embodiment 的 action lag、hard interval 比例和 active-slot coverage。
4. 建立跨数据集视觉/轨迹近重复索引。
5. 基于 train/A/joint case 重建 normalization stats。
6. 更新 FastWAM Stage 1 balanced sampler 配额。
7. 使用新 stats 重跑 Stage 2 memory optimizer/checkpoint/resume。
8. 对多 segment `action_config` 增加显式 boundary-aware window/prompt 策略。

在这些任务完成前，文档应表述为“适配器和真实样本 TrainingCase 已调通”，不应表述为
“LingBot-VA/DreamZero 全量 FastWAM 预训练已完成”。
