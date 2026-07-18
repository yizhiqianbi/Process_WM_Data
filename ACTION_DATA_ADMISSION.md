# FastWAM 动作数据准入、映射与实测报告

更新日期：2026-07-18

本文回答两个问题：各数据集为什么此前没有进入 action 训练，以及现在什么数据能够在有证据的前提下进入 FastWAM Stage 2。结论基于官方格式说明、下载到本机的真实文件和 81 帧端到端回归，不以目录名或向量维度猜测动作语义。

## 1. 当前结论

七个数据集的当前真实样本都已生成 `joint_video_action` case；这表示逐库接入路径通过，
不表示全量数据已经完成下载和清洗：

| 数据集 | 当前 action 状态 | 当前验证范围 | 81/80 实测 |
| --- | --- | --- | --- |
| OXE | 部分开放 | `asu_table_top_converted_externally_to_rlds` | 通过 |
| OXE-AugE | 开放派生 target | 每个 target robot 独立 episode | 通过 |
| AgiBot-Beta | 开放 | 真实 task 389 / episode 673828 | 通过，18 个窗口 |
| RoboCOIN | 开放 | 本地 `AI2_Alphabot_2` 样本 | 通过 |
| RoboMIND | 按官方本体表开放 | 本地 `h5_ur_1rgb` 样本 | 通过 |
| Galaxea | 开放 | 本地 `r1lite` LeRobot 样本 | 通过 |
| InternData-A1 | 开放 | 本地 `a2d` LeRobot 样本 | 通过 |

这里的“通过”表示真实源文件完成 `scan -> clean -> materialize -> windows -> cases`，并至少产出一个包含 81 个 canonical state 点、80 个 canonical action transition 和 21 个视频采样点的训练 case。它不表示全量下载已经完成，也不表示所有本体和子数据集都已自动获得 action 权限。

## 2. Action 准入硬条件

一个 episode 只有同时满足以下条件，才允许 `action_loss_mask=true`：

1. 原始控制文件可由受支持的结构化 reader 解码，不能只依据 metadata 推断。
2. state 和 action 的每个激活维度都有明确源 key、源 index、canonical slot 和语义。
3. mapping 没有 slot collision，且 adapter 将其标记为 `verified=true`。
4. action 类型明确，例如 joint target、EEF target、velocity 或派生的 next-state target。
5. timestamp 或可信的固定 FPS 可建立单调控制时间线，时长至少覆盖 4 秒。
6. 激活信号 finite，shape 和 episode 长度一致。
7. state/action 公共且可比的 canonical 维度通过 alignment 检查。
8. 当前 81-step window 不与 hard temporal、state 或 action bad interval 相交。
9. 视频满足 Stage 1 的结构和最小独立帧覆盖要求。

任何一项失败都不会填零冒充 action。管线会保留视频可用样本，并把训练模式降为 `video_only`；loader 同时输出全 invalid action mask。

## 3. 统一 canonical 合同

80D canonical 向量按固定 block 分配：

| Slot | 内容 |
| --- | --- |
| `0:6` | left/primary EEF：xyz + rotation vector |
| `6` | left/primary gripper |
| `7:13` | right EEF：xyz + rotation vector |
| `13` | right gripper |
| `14:21` | left/primary arm joints 1-7 |
| `21:28` | right arm joints 1-7 |
| `28:40` | left hand joints，最多 12 维 |
| `40:52` | right hand joints，最多 12 维 |
| `52:58` | mobile base linear/angular 6D |
| `58:64` | waist/torso/head，最多 6 维 |
| `64:80` | reserved，始终 invalid |

不同本体不需要填满 80D。有效性由 `state_dim_valid_mask` 和 `action_dim_valid_mask` 决定，数值 0 本身不代表该维存在。单臂机器人统一占 primary/left block，但 provenance 会保留真实机器人名称。

控制时间线重采样到 20 Hz：

```text
state:   t[0] ... t[80]       81 points
action:  a[0] ... a[79]       80 transitions
video:   offsets 0,4,...,80    21 points
duration: 4 seconds
```

重采样前保留原始时间戳和 nearest-source index。训练侧使用前 80 个 state 作为 proprio，第 81 个 state 用于定义最后一个 transition 的终点。

## 4. 原生数据读取层

统一 reader 当前支持：

- 本地 Parquet 和 `tar://archive!member.parquet`；
- OXE tar 内受限 pickle，只读取数组、标量和图像字段；
- 本地 HDF5 和 `tar://archive!member.h5`；
- HDF5 纳秒时间戳归一化为秒；
- 派生列 `pose_rpy_to_rotvec`；
- 派生列 `next_row_hold_last`，用于明确标注的 next-state target。

reader 只解决“能否正确取数”，不自动授予 action 权限。权限仍由 adapter 的 embodiment-specific contract、mapping provenance 和 cleaner 的实测结果共同决定。

## 5. 分数据集分析

### 5.1 OXE / Open X-Embodiment

**此前未进入 action 的原因**

OXE 是许多来源数据集的集合。官方统一格式给出 7D EEF action 外形，但每个来源的 absolute/delta/velocity、坐标系、控制频率和 state key 仍可能不同。给 55 个来源套一个猜测 mapping 会把不等价的动作混在同一 normalization domain 中。

**现在开放的范围**

当前只为 `asu_table_top_converted_externally_to_rlds` 建立了严格 contract：

| canonical | 原始字段 | 变换 |
| --- | --- | --- |
| state `0:6` | `ground_truth_states.EE[0:6]` | xyz 保留，rpy 转 rotation vector |
| state `6` | `observation.state[6]` | gripper |
| action `0:6` | `action[0:6]` | xyz 保留，rpy 转 rotation vector |
| action `6` | `action[6]` | gripper target |

官方 ASU schema 将 `ground_truth_states.EE` 定义为 xyz+rpy，并将 observation state 定义为 6 个关节加 gripper。本地真实轨迹还验证了 `action[t,0:6]` 与下一时刻 EE state 的数值关系，因此这里记录为 `native_next_eef_pose_and_gripper_target`，不是凭 7D shape 猜测。

源频率按该子集 5 Hz 读取，再重采样到 20 Hz。非 ASU OXE episode 现在可以原生解码和执行视频/信号审计，但在逐子集 contract 建立前仍为 `video_only`。

官方依据：

- [Open X-Embodiment 官方仓库](https://github.com/google-deepmind/open_x_embodiment)
- [ASU Table Top TFDS schema](https://www.tensorflow.org/datasets/catalog/asu_table_top_converted_externally_to_rlds)

### 5.2 OXE-AugE

**此前未进入 action 的原因**

release 中没有一个可以直接当作统一控制命令的 `action` 列，而且一个源 episode 同时包含多个 target robot 的 replay 轨迹和视频。此前若把这些机器人当成多相机，会错误地把不同本体的画面和关节向量放进一个训练样本。

**当前处理**

根据官方 AugE replay 设计，每个 target robot 被拆成独立 episode：

```text
state[t]  = observation.<target_robot>.joints[t]
action[t] = observation.<target_robot>.joints[t+1]
terminal  = hold_last
```

这类 action 明确标记为 `derived_next_replay_joint_target`，不是机器人记录的原生硬件 command。它适合训练“给定当前 replay state 预测下一 target state”，不能在论文统计中与 native control command 混称。

7D 或 8D replay vector 被解释为 6/7 个 arm joint 加 1 个 gripper，映射到 `14:20/21` 和 slot `6`。每个 episode 只引用该 target robot 的对应视频。所有 target variant 继承同一个 source lineage，避免原 episode 与增强结果跨 train/validation 泄漏。

真实扩展回归扫描了 9 个 variant，得到 9 个 action-eligible episode、96 个 action window。第一个源 family 覆盖 `google_robot`、`jaco`、`kinova3`、`kuka_iiwa`、`panda`、`sawyer`、`widowX` 和 `xarm7`。

官方依据：[AugE Toolkit](https://github.com/BerkeleyAutomation/AugE-Toolkit)

### 5.3 AgiBot-Beta

**当前开放状态**

从正在续传的首个官方 `proprio_stats` 分片中，已经读取到完整真实 member
`389/673828/proprio_stats.h5`，并与本地 observation tar 和 `task_info/task_389.json` 完成
episode ID join。真实链路已执行 `scan -> clean -> materialize -> windows -> cases -> FastWAM
loader`，不再依赖合成 HDF5 推断 action 可用性。

`parameters/` 只提供相机内外参。缺少它会令 case 中的 calibration availability 为 false，
但不会阻断控制时间线、state/action 清洗或 Stage 2 action loss。

**HDF5 字段和 canonical mapping**

| HDF5 state/action pair | 维度 | canonical |
| --- | ---: | --- |
| `state/joint/position` / `action/joint/position` | 14 | left/right arm `14:28` |
| `state/effector/position` / `action/effector/position` | 2 | gripper `6,13` |
| `state/head/position` / `action/head/position` | 2 | `60,61` |
| `state/waist/position` / `action/waist/position` | 2 | `58,59` |

当前实样激活 20 个 state/action slot：

```text
6, 13, 14..27, 58, 59, 60, 61
```

时间戳按纳秒解析。官方将 state 描述为传感器/执行器 feedback，将 action 描述为发送给
HAL 的 command。effector feedback 当前实样约为毫米范围，而 command 为 `[0,1]` 闭合量，
所以 gripper slot 标记为 `alignment_safe=false`：它们仍参与训练和各自 domain 的归一化，
但不会被错误的同量纲 alignment 假设阻断。双臂、waist 和 head 共 18 个可比 slot 的
alignment score 为 1.0，最佳 lag 为 0。

**有效 action 索引**

episode 原始数组有 1226 行，但各 action group 含独立 `index` 数据集。joint action 的
有效范围为原始 index `24..1190`，共 1167 行；reader 对所有存在的 action index 集合取
交集，再按原始 index 同步筛选 timestamp、state 和 action。`frame_index` 保留原始编号，
时间戳从第一个有效行归零。不能直接截取数组前 1167 行，否则会让控制和视频错位 24 帧。

**真实回归结果**

筛选后的 1167 行约为 29.986 Hz，重采样到 20 Hz 后产生 786 个 canonical state 点和
785 个有效 transition。以 81 个 state、80 个 action、stride 40 构造 18 个窗口，18/18
都通过视觉、时间、signal 和 alignment 清洗，最终为 A 级 `joint_video_action`。三路相机
为 `head_color`、`hand_left_color`、`hand_right_color`。

训练 loader 实测输出 `video [3,21,384,320]`、`action/proprio [80,80]`，20 个维度有效，
8/2/1 memory 的 11 个历史索引均严格早于当前窗口。完整说明见
`AGIBOT_PROPRIO_TRAINING.md`。

官方依据：[AgiBot World](https://github.com/OpenDriveLab/AgiBot-World)

### 5.4 RoboCOIN

RoboCOIN 原本就是当前唯一已开放 action 的基线。LeRobot `action` 和 `observation.state` 使用带单位和部件名的 feature schema。当前本地 `AI2_Alphabot_2` 样本激活 20 个 slot：

- left/right 7DoF arm：`14:28`；
- left/right gripper：`6,13`；
- torso/head：`58:62`。

EEF position 等 frame-sensitive state 字段仍保留在 native schema，但自动 mapping 将中等置信度字段设为 inactive，不会在没有坐标系 contract 时混入 action loss。全量训练前仍需按 robot/config 分 normalization domain，并重新计算 train-only statistics。

### 5.5 RoboMIND

**此前未进入 action 的原因**

HDF5 同时存在 `master` 和 `puppet` 轨迹，不同本体的官方训练配置选择不同的 action root。简单假设“master 永远是 action”在 UR、Franka 和部分 TienKung 数据上是错误的。

**当前官方 contract 表**

| embodiment | state root | action root | control key | layout |
| --- | --- | --- | --- | --- |
| `h5_franka_3rgb` | puppet | puppet | `joint_position` | single + gripper |
| `h5_ur_1rgb` | puppet | puppet | `joint_position` | single + gripper |
| `h5_agilex_3rgb` | puppet | master | `joint_position_left/right` | dual split + gripper |
| `h5_tienkung_gello_1rgb` | puppet | master | `joint_position` | dual packed + gripper |
| `h5_tienkung_xsens_1rgb` | puppet | puppet | `joint_position` | dual packed, no gripper |

只有命中表格、控制 key 存在、state/action 维度一致的 HDF5 才建立 verified mapping。其他 RoboMIND embodiment 保持视频-only，直到官方配置补充。当前本地 `h5_ur_1rgb` 实样使用 `puppet/joint_position` 的 6 个关节加 gripper，映射到 `14:20` 和 `6`，已经完成原生 HDF5 读取、embedded image 解码和 action window 回归。

颜色通道也按官方配置区分，Franka/UR 的特定图像键执行 BGR 到 RGB；其他本体不盲目转换。

官方依据：

- [RoboMIND dataset card](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)
- [RoboMIND dataset utils](https://github.com/Open-X-Humanoid/RoboMIND-dataset-utils)

### 5.6 Galaxea

**此前未进入 action 的原因**

旧 cleaner 对所有可见 action telemetry 做 episode 级硬阻断，其中未进入 canonical mapping 的 EEF/quaternion 监控字段也会触发 abrupt warning；同时单个 quaternion 的 `q` 与 `-q` 等价没有被正确处理。这会把可用 joint target 错判为坏 action。

**当前处理**

tar 内 LeRobot Parquet 和 MP4 直接读取。只对 canonical-active control signal 执行 action 准入阻断，未映射的 EE telemetry 仍保留统计但只作为 monitor。四元数使用归一化和 geodesic step，不再按四个普通标量判断跳变。

当前 `r1lite` 实样激活：

- 双臂 joint target 和双 gripper；
- chassis 6D target twist；
- torso 6D target twist。

state/action 公共关节和 gripper slot 用于 alignment；只有 action 侧存在的 velocity target 不强制寻找同 slot feedback。实测得到 24 个 action window。

### 5.7 InternData-A1

**此前未进入 action 的原因**

旧 LeRobot schema 选择器可能在已有官方 `actions.*` 时仍混入 `master_actions.*`，造成语义碰撞；同时 `left_arm_0`、`right_arm_0`、裸 `left_gripper`，以及 waist/head 的简写名字未被 strict mapper 识别。

**当前处理**

schema 优先选择官方 `action`/`actions.*`，仅在两者均不存在时回退 `master_actions.*`。结合 source key 解析零起始维度名：

- `actions.joint.position[0:14]` -> 双臂 `14:28`；
- `actions.effector.position[0:2]` -> gripper `6,13`；
- `actions.waist.position[pitch,lift]` -> `58,59`；
- `actions.head.position[yaw,patch]` -> `60,61`，其中 `patch` 保留 release 中的拼写但解释为 pitch。

对应 observation state 使用相同 slot。当前本地 `a2d` 实样 20 个 action slot 全部映射且 alignment 通过，得到 8 个 action window。`real`、`sim` 和 `sim_updated` 后续仍必须分开建立 normalization domain。

官方依据：[InternData-A1 dataset card](https://huggingface.co/datasets/InternRobotics/InternData-A1)

## 6. 时序和异常动作清洗

动作跳变分三层处理：

1. 超过 robust threshold 的普通 abrupt step 记 soft warning。
2. 超过 `10 x robust threshold` 的 step 建立局部 hard interval，只降级与它相交的 81-step window。
3. 只有普通 abrupt ratio 大于 0.20，并且 extreme abrupt ratio 同时大于 0.05，才把整个 episode 阻断。

这样可以保留真实高速操作，也不会让少量损坏 transition 污染 action loss。OXE ASU 实测 34 个候选 window 中 28 个保留 action，6 个与 hard interval 相交后降为 video-only；其他已开放数据集的当前回归没有发生 window downgrade。

alignment 只使用同时存在于 state/action 且 `alignment_safe=true` 的 canonical slot。不同单位或不同物理量不会因为 slot 名相近而强行计算相关性。派生 next-state target 的 provenance 会进入 manifest，便于训练时分 domain 或单独设置 loss weight。

## 7. 真实回归

命令：

```bash
PY=${PYTHON_BIN:-python3}
cd /path/to/Process_WM_Data

$PY scripts/run_pipeline.py \
  --datasets all \
  --output-root work/validation/action_recovery_all \
  --max-episodes 1 \
  --num-frames 81 \
  --target-fps 20 \
  --verify-files \
  --check-videos \
  --workers 7
```

结果：

| 数据集 | episode | video windows | action windows | grouped joint cases | grouped video-only cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| OXE ASU | 1 | 34 | 28 | 3 | 2 |
| OXE-AugE | 1 | 11 | 11 | 1 | 0 |
| AgiBot-Beta | 1 | 20 | 0 | 0 | 1 |
| RoboCOIN | 1 | 15 | 15 | 1 | 0 |
| RoboMIND | 1 | 2 | 2 | 1 | 0 |
| Galaxea | 1 | 24 | 24 | 1 | 0 |
| InternData-A1 | 1 | 8 | 8 | 1 | 0 |

OXE-AugE 额外执行 `max-episodes=9` 回归，9/9 episode 进入 action，产生 96 个 action window 和 9 个 grouped joint cases。

2026-07-18 追加 AgiBot proprio 实样回归：task 389 / episode 673828 产生 18 个 video
window 和 18 个 action window，1 个 grouped joint case，0 个 video-only case。该结果位于
`work/stage_pipeline/agibot_beta/`，不会覆盖上表保留的 2026-07-17 observation-only 基线。

机器可读结果位于：

```text
work/validation/action_recovery_all/<dataset>/pipeline_summary.json
work/validation/action_recovery_all/<dataset>/clean/episodes.cleaned.jsonl
work/validation/action_recovery_all/<dataset>/cases/training_cases.jsonl
work/validation/action_recovery_v2/oxe_auge/
```

自动测试当前为 33 项，覆盖 OXE 派生变换、HDF5 纳秒时间戳、AgiBot tar offset 读取与
valid-index 交集、InternData 命名、局部 bad interval、normalization 过滤和完整 pipeline contract。

## 8. 代码位置

| 功能 | 路径 |
| --- | --- |
| 80D mapping 与 verified provenance | `fastwam_preprocess/canonical.py` |
| Parquet/OXE pickle/HDF5 reader | `fastwam_preprocess/source.py` |
| action alignment 和 bad interval | `fastwam_preprocess/cleaning.py` |
| signal audit | `fastwam_preprocess/signal_audit.py` |
| OXE contract | `fastwam_preprocess/adapters/oxe.py` |
| OXE-AugE target variants | `fastwam_preprocess/adapters/oxe_auge.py` |
| AgiBot observation/proprio join | `fastwam_preprocess/adapters/agibot.py` |
| RoboMIND official embodiment table | `fastwam_preprocess/adapters/robomind.py` |
| LeRobot schema selection | `fastwam_preprocess/adapters/lerobot.py` |
| cleaning thresholds | `configs/cleaning_policy_v1.yaml` |

## 9. 仍需完成的工作

1. 续传 AgiBot 剩余六个 `proprio_stats` tar 和全部 `task_info`，再执行全量 join/size 验证。
2. 为 OXE 其余子数据集逐个建立 action type、frame、frequency 和 mapping contract，不能从 ASU 外推。
3. 扫描 RoboMIND 全量 embodiment 命中率；未列入官方 contract 表的本体继续保持 B 级。
4. 对七个已开放样本路径运行全量 manifest，统计 hard interval、alignment lag 和 action slot 覆盖率。
5. 用全量 train split 重建每个 embodiment 的 normalization statistics。
6. 更新 FastWAM Stage 2 多数据集 stats 后，再做一次七域真实 optimizer/checkpoint/resume 回归。

最后一项尚未执行。当前已经证明的是七个数据集的样本级处理路径能产出统一 action case，
不应把它表述为七域 FastWAM 正式预训练已经完成。
