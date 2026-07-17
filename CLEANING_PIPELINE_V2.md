# FastWAM Cleaning Pipeline V2

本文档描述 `Preprocess_FastWAM` 当前可执行的第二版数据清洗设计。它参考：

`../World+Action+Model+-+Data+Statistics_(1)(1).pdf`

目标不是把 PDF 中的经验阈值原样复制，而是把其中的三级清洗思想改造成适用于 OXE、
OXE-AugE、AgiBot-Beta、RoboCOIN、RoboMIND、Galaxea 和 InternData-A1 的统一、可审计、
可分阶段训练的数据契约。

## 1. 结论

V2 采用下面五条原则：

1. **episode 是审计边界，window 是训练准入边界。** 一个坏点不应默认报废整条长轨迹。
2. **视频、状态、动作、语言分别评分。** 不再按 warning 数量线性扣分。
3. **语义优先于逐维数值。** 四元数用旋转几何检查，周期角先 unwrap，不能把表示等价误判为跳变。
4. **stage 1、stage 2、stage 3 使用不同准入条件。** 没有可靠动作的数据仍可用于 video backbone。
5. **硬规则只处理确定性错误。** 模糊、静止、低分辨率、低 FPS、VLM 判断等默认是软信号，必须先校准。

当前数据流为：

```text
raw dataset
  -> scan/episodes.jsonl
  -> clean/episodes.cleaned.jsonl + cleaning_report.jsonl
  -> canonical/canonical_episodes.jsonl + canonical.parquet
  -> windows/windows.jsonl
  -> cases/training_cases.jsonl
```

V2 新增字段会贯穿 `clean -> materialize -> windows -> TrainingCaseV1`，原有训练字段仍保留。

## 2. PDF 建议如何落地

### 2.1 已直接采用

| PDF 建议 | V2 实现 |
| --- | --- |
| 每条 episode 保存路径、检查结果和过滤原因 | `cleaning_report.jsonl` 保存完整审计；cleaned manifest 保存压缩摘要 |
| NaN/Inf、长度、时间轴、静态信号检查 | 结构、时序、逐信号审计已实现 |
| 加速度、jerk 和跳变检查 | 每维输出 velocity/acceleration/jerk 鲁棒统计 |
| 首尾/稀疏视觉质量检查 | 每路选中相机默认均匀抽 9 帧 |
| 黑白帧、亮度、模糊、冻结检查 | luma 分位数、黑白像素比例、entropy、Laplacian、帧差和 dHash |
| 统一相机名、频率、动作空间、归一化域 | 五相机槽、20 Hz、80D canonical + mask、normalization domain |
| 数据平衡和去重 | 输出 trajectory/visual fingerprint、lineage-safe split、sampling weight |

### 2.2 调整后采用

**低分辨率不做统一硬切。** PDF 中的 224/336 像素建议适合特定视觉 backbone，但 OXE 中存在有效的
低分辨率源。V2 记录分辨率，是否 resize 由训练 profile 决定。

**低 FPS 不以 5 FPS 一刀切。** FastWAM 的 4 秒窗口需要 21 个视觉采样点，但源视频可以低帧率。
V2 使用“预计独立源帧数”准入，默认至少 8 帧。2 FPS 视频在 4 秒内约有 9 个独立帧，仍可保留；
约 1 FPS 的视频默认拒绝该窗口。

**模糊和冻结默认是软标记。** 机器人停顿、相机随末端运动、景深变化都可能产生低帧差或低
Laplacian。只有连续极黑/极白样本形成 hard visual interval；模糊、低 entropy 和疑似冻结先降低权重。

**动作跳变分两级。** 鲁棒阈值以上记为 soft abrupt；超过阈值 10 倍才形成 hard interval。
没有机器人本体物理上限时，不把正常的高速动作切换误删。

### 2.3 暂不作为硬规则

- `CLIP(end, text) - CLIP(start, text) > 0`：搅拌、擦拭、循环操作和视角移动不满足该单调假设。
- 单次 VQA/VLM 成功判断：在人工标定精度前只适合输出 soft confidence。
- 自动 re-caption 覆盖原指令：原始语言必须保留；新 caption 应作为并列字段并记录模型版本。
- 只按全局百分位删除轨迹：不同 embodiment、控制模式和单位不能共用一个全局动作阈值。

## 3. 清洗级别

### 3.1 Level 0: 下载和文件完整性

由 downloader 和 adapter 的 `scan` 阶段完成：

- 固定 Hugging Face revision；
- 检查 partial/incomplete 文件；
- 建立 episode 边界；
- 检查引用的 Parquet、视频、tar member 或 HDF5 是否存在；
- 保存 dataset、release、embodiment、task、lineage 和原始 URI。

目录存在或文件体积较大不能替代完整性检查。

### 3.2 Level 1: 元数据和信号清洗

`clean` 对统一 reader 可读的 Parquet、OXE pickle 和 HDF5 执行：

- row count 与 manifest 长度一致性；
- timestamp finite、严格递增、gap、jitter 和 declared FPS 偏差；
- frame index 连续性；
- signal 行宽、finite ratio、常量维；
- 每维 p01/p50/p99、range、step median/MAD、最大跳变；
- velocity、acceleration、jerk 的 p50/p99/max 和鲁棒异常点；
- action-state 直接反馈或 action-vs-state-derivative 相关性和 lag；
- 任务文本去空白、去重和可用性。

Native 源采取同一套分层策略：reader 可解码只证明数据可审计，不自动证明 action 可训练。
OXE pickle、AgiBot tar 内 HDF5 和 RoboMIND HDF5 只有命中 embodiment-specific contract、
verified canonical mapping、时间线和 alignment 门控后才能进入 action；其余数据仍可独立完成
视觉检查并进入 Stage 1。

### 3.3 Level 2: 稀疏视觉清洗

执行 `clean --check-videos` 时，默认按训练相机角色选择 3 路：

1. `global_primary`
2. `left_wrist`
3. `right_wrist`
4. 若缺失则依次补 `global_secondary`、`auxiliary` 或其他相机

每路默认均匀抽取 9 帧并缩放为 64x64 灰度审计图，只用于质量统计，不作为训练图像。计算：

- mean luma、p01/p50/p99；
- 像素值 `<=10` 和 `>=245` 的比例；
- 16-bin normalized entropy；
- normalized Laplacian variance；
- 相邻稀疏帧 mean absolute difference；
- 64-bit dHash distance；
- 稀疏视觉 fingerprint。

支持的源：

| 存储 | 支持情况 |
| --- | --- |
| 本地 MP4/MKV/AVI/MOV | ffprobe + ffmpeg 稀疏解码 |
| `tar://archive!member.mp4` | 同 archive 多相机一次打开，逐 member 临时物化后审计 |
| `hdf5://file#dataset` | 直接稀疏读取内嵌 JPEG/array |
| `oxe-pickle://archive!member#observation.key` | restricted pickle reader 后稀疏读取图像 |
| 未知 URI | `pending`，不伪装成 passed |

`--decode-videos` 会额外对选中视频做完整 ffmpeg decode。它是高 I/O 深审计，不应和第一次全量 scan 同时运行。

### 3.4 Level 3: 语义/几何清洗

#### 四元数

从 feature names 中识别 `orientation.{x,y,z,w}`、`quaternion` 或 `quat` 组：

- 检查 norm；
- 使用 `q` 与 `-q` 等价性做符号连续化；
- 用 `2*acos(abs(dot(q_t, q_t+1)))` 计算 geodesic rotation；
- 四元数分量不再参与普通逐维 abrupt hard decision；
- 符号连续化只用于审计，不静默覆盖原始源数据。

这修复了 Galaxea EE pose 中“分量变化看似很大、实际旋转 geodesic 很小”的误报。

#### 周期角

名称明确包含 `rad`、`euler`、`yaw`、`pitch`、`roll` 的位置量在审计前 unwrap。
velocity/twist 不做 unwrap。

#### 机器人物理上限

当前代码输出可用于标定的速度、加速度和 jerk 指标，但默认没有为七个数据集写死统一物理上限。
下一步应按 `dataset/embodiment/signal semantic` 配置 joint limit、max velocity、max acceleration、
workspace 和 gripper 范围。没有单位和控制语义确认前，禁止把此类阈值设成全局 hard rule。

## 4. `bad_intervals` 契约

每个局部异常被压缩成 half-open interval：

```json
{
  "timeline": "canonical_20hz",
  "start": 100,
  "stop_exclusive": 103,
  "start_s": 5.0,
  "stop_s": 5.15,
  "reason": "extreme_abrupt_signal_step",
  "domains": ["action"],
  "severity": "hard",
  "source_key": "action.right_arm"
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `timeline` | `source_control`、`source_video` 或物化后的 `canonical_20hz` |
| `start/stop_exclusive` | half-open 帧区间 |
| `start_s/stop_s` | 时间区间，用于跨频率映射 |
| `domains` | `video`、`temporal`、`state`、`action` 中一个或多个 |
| `severity` | `hard` 参与窗口剔除；`soft` 仅用于评分、采样和复查 |
| `reason` | 稳定的机器可读原因 |
| `source_key/camera_key` | 原始 signal 或相机来源 |

当前 hard interval：

- non-finite signal；
- 超过鲁棒 abrupt threshold 10 倍的极端跳变；
- non-finite 或零范数 quaternion；
- timestamp gap；
- 连续两个以上稀疏样本极黑或极白。

当前 soft interval：

- 普通 robust abrupt；
- acceleration/jerk robust outlier；
- 超出容差的 quaternion norm；
- rapid quaternion rotation；
- 单个极黑/极白样本；
- low entropy、low sharpness；
- possible frozen video。

Materialize 会使用时间戳把 source interval 映射到 20 Hz canonical timeline，并保留 source interval 字段。

## 5. 窗口级准入

FastWAM 默认窗口：

- 20 Hz control timeline；
- 81 个 state point；
- 80 个 action transition；
- 4 秒；
- video offsets `0,4,...,80`，共 21 点；
- 默认 window stride 40。

窗口生成逻辑：

```text
all starts
  - overlap hard(video|temporal)             -> reject window
  - overlap hard(state|action)               -> video_only
  - no hard overlap + episode action verified -> joint_video_action
```

两类输出互不重复。一个 episode 可以同时输出多个 `joint_video_action` range 和多个 `video_only` range。
不连续 start 会拆成多个紧凑 `valid_starts` 记录，旧 loader 仍按下面字段读取：

```json
{
  "valid_starts": {
    "start": 120,
    "stop_exclusive": 240,
    "stride": 40,
    "count": 3
  }
}
```

V2 summary 额外报告：

- `interval_filtered_video_window_count`
- `interval_downgraded_action_window_count`
- `low_unique_frame_coverage_episode_count`
- `video_window_count`
- `action_window_count`

## 6. 分维度质量分数

`quality.component_scores` 包含：

| 分量 | 权重 | 主要来源 |
| --- | ---: | --- |
| `integrity` | 0.25 | source、structure、readability |
| `temporal` | 0.20 | timestamp、gap、jitter、FPS |
| `visual` | 0.20 | container/decode 和 sparse quality |
| `kinematic` | 0.20 | signal、semantic geometry、alignment |
| `language` | 0.10 | prompt 可用性 |
| `novelty` | 0.05 | fingerprint dedupe 状态 |

总分是可用分量的加权平均，不再因为相机多、signal 多而按 warning 条数重复扣分。

同时保留：

- `quality.hard_blockers`
- `quality.soft_flags`
- `quality.bad_intervals`
- `quality.sampling_weight`
- 兼容字段 `tier/candidate_tier/video_eligible/action_eligible`

Tier 只表达能力边界：

- **A**：视频、状态、动作、canonical mapping、时序和 alignment 已验证；
- **B**：视频可用，但 action loss 必须关闭或仍待语义验证；
- **C**：源损坏、空 episode、所有视觉源失败或其他确定性 hard failure。

## 7. 三阶段训练准入

### Stage 1: video backbone

要求：

- episode/video 可读；
- 至少覆盖一个 4 秒窗口；
- 预计独立源视频帧数默认不少于 8；
- 窗口不与 hard video/temporal interval 重叠。

不要求 state/action/canonical action mapping。OXE、AgiBot observation、RoboMIND 等可以先进入此阶段。

### Stage 2: memory FastWAM

要求：

- stage 1 条件满足；
- state/action schema 和 80D mask 已生成；
- canonical action mapping verified；
- action-state alignment passed；
- 窗口不与 hard video/temporal/state/action interval 重叠；
- 81/80/21 timeline contract 完整。

### Stage 3: target dataset fine-tuning

首先必须是 stage 2 candidate，再按目标数据集增加：

- success/failure 标签策略；
- 目标 robot 的物理范围；
- task/scene/operator 泄漏检查；
- 目标域 normalization stats；
- 目标任务采样和平衡策略。

`stage_admission` 在 episode、window 和 TrainingCase 中均可读取。Stage 3 的
`target_dataset_and_success_policy_required` 是显式待办，不会被误认为已经验证。

## 8. 去重与数据泄漏

当前单次 `clean_manifest` 运行输出：

- `source_uri_sha256`：源 URI 身份，不是文件内容 hash；
- `trajectory_sha256`：最多 32 个均匀控制样本、按 signal key 排序和数值量化后的 fingerprint；
- `visual_sha256`：选中相机稀疏 dHash 序列的组合 fingerprint；
- `dedupe.status/group_id/duplicate_of`；
- 基于 `lineage_id` 或稳定 source identity 的 `split_group_id`。

单次 clean 中 dedupe fingerprint 的优先级为
`trajectory fingerprint > visual fingerprint > source URI`。训练 split 当前使用
`lineage_id > source identity`，因此可选的视觉审计开关不会改变 split。

限制：当前 dedupe 状态只在一次 clean invocation 内比较，尚不会自动重写 split group。七数据集全局精确/近似去重仍应增加一个独立汇总阶段，
把所有 cleaned manifest 的 fingerprint 合并后再冻结 train/validation/test split。尤其需要把 OXE-AugE 原图、
机器人替换视图、inpainting variant 和 OXE 源 episode 绑定到同一 lineage group。

## 9. 七数据集覆盖

| 数据集 | Level 1 | Level 2 | Action 训练现状 |
| --- | --- | --- | --- |
| OXE | pickle 数组和派生列已支持 | pickle 内嵌图像已支持 | ASU UR5 已进入 A；其他子集仍逐项验证 |
| OXE-AugE | target-robot replay Parquet 已支持 | target robot 本地 MP4 已支持 | next replay state 作为有 provenance 的派生 target；9 个 variant 回归通过 |
| AgiBot-Beta | observation/proprio tar join 已实现 | tar 内 MP4 已支持，同 tar 合并打开 | 当前下载缺 `proprio_stats`，真实样本仍为 B |
| RoboCOIN | Parquet、signal、alignment 已支持 | 多路 MP4 已支持 | 当前实样可进入 A；需按 embodiment 校准 |
| RoboMIND | 官方本体表驱动的 HDF5 state/action reader | HDF5 内嵌 JPEG/array 已支持 | 表内本体可进入 A；未知本体保持 B |
| Galaxea | tar 内 Parquet 和 canonical-active signal audit | tar.gz 内 MP4 已支持 | R1-lite 实样进入 A，24 个 action window |
| InternData-A1 | LeRobot 原生 `actions.*` 和 state 已支持 | 多路 MP4 已支持 | A2D 实样 20D mapping/alignment 通过，进入 A |

“支持视觉审计”不等于“全量下载完成”，也不等于“action 语义已验证”。三种状态必须分开记录。

## 10. 默认阈值

权威配置：`configs/cleaning_policy_v1.yaml`

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `min_frames` | 81 | 最小控制点数 |
| `minimum_finite_ratio` | 1.0 | action/state finite 要求 |
| `abrupt_warning_ratio` | 0.02 | 逐 signal soft warning 比例 |
| `abrupt_action_block_ratio` | 0.20 | episode action 全局阻断比例 |
| `extreme_abrupt_action_block_ratio` | 0.05 | 与普通 abrupt 比例同时超限才全局阻断 |
| `extreme_abrupt_multiplier` | 10 | hard local jump 相对鲁棒阈值倍数 |
| `bad_interval_padding_frames` | 1 | 局部区间前后扩张 |
| `quaternion_norm_tolerance` | 0.05 | quaternion norm 容差 |
| `quaternion_max_step_rad` | 1.5 | 单步 geodesic soft warning |
| `sparse_visual_sample_count` | 9 | 每路相机稀疏样本数 |
| `sparse_visual_max_cameras` | 3 | 每 episode 默认审计相机数；`<=0` 表示全相机 |
| `visual_extreme_pixel_ratio` | 0.98 | 极黑/极白像素比例 |
| `visual_minimum_entropy` | 0.08 | normalized entropy soft threshold |
| `visual_minimum_laplacian_variance` | 0.0005 | normalized sharpness soft threshold |
| `visual_freeze_mean_absolute_difference` | 0.002 | 相邻样本冻结候选阈值 |

这些是保守初值，不是最终统计结论。阈值版本必须跟训练 manifest 一起冻结。

## 11. 推荐运行流程

### 11.1 第一遍：不解码视频

```bash
cd /path/to/Process_WM_Data

python3 scripts/run_pipeline.py \
  --datasets all \
  --workers 2 \
  --verify-files \
  --cleaning-policy configs/cleaning_policy_v1.yaml
```

这一遍用于发现结构、下载、schema、timestamp、signal 和 mapping 问题。

### 11.2 分层视觉校准

先为每个 dataset/embodiment/task 抽 100-1000 条，而不是立刻对全部视频 full decode：

```bash
python3 -m fastwam_preprocess.cli clean \
  --manifest work/v1/robocoin/scan/episodes.jsonl \
  --output-root work/audit_calibration/robocoin \
  --max-episodes 500 \
  --check-videos \
  --policy configs/cleaning_policy_v1.yaml
```

检查 luma、entropy、Laplacian、frame-diff 和 bad interval 的分布及样本画面，再按 embodiment 调整阈值。

### 11.3 深视频审计

仅对结构已通过、准备进入训练冻结版本的数据执行：

```bash
python3 -m fastwam_preprocess.cli clean \
  --manifest work/v1/robocoin/scan/episodes.jsonl \
  --output-root work/deep_video_audit/robocoin \
  --check-videos \
  --decode-videos \
  --policy configs/cleaning_policy_v1.yaml
```

tar.gz 随机读取和 full decode 都是高 I/O 操作，应与 downloader 错峰，并限制 dataset-level worker。

### 11.4 独立重建窗口

阈值变化后可以从 canonical manifest 单独重建窗口：

```bash
python3 -m fastwam_preprocess.cli windows \
  --manifest work/v1/robocoin/canonical/canonical_episodes.jsonl \
  --output-root work/v1/robocoin/windows \
  --num-frames 81 \
  --stride 40 \
  --action-video-freq-ratio 4 \
  --minimum-unique-video-frames 8
```

## 12. 阈值校准方法

每个 dataset 至少按以下 strata 抽样：

- embodiment/robot type；
- task namespace；
- camera role；
- episode length bucket；
- source FPS/resolution bucket；
- quality tier；
- success/failure（若有）；
- source/augmentation lineage。

每个候选 hard rule 需要报告：

1. 规则命中 episode 数和 window 数；
2. 按 dataset/embodiment/task 的命中分布；
3. 人工复核 precision；
4. 对 stage 1/2/3 各自保留的小时数；
5. train/validation/test lineage 泄漏检查；
6. 阈值变化前后的训练 loss、validation rollout 或 proxy metric。

建议 hard rule 的人工 precision 达到 99% 左右再用于不可逆删除；否则保留源数据，只在 manifest 中降权或屏蔽窗口。

## 13. 当前真实回归结果

2026-07-17 在本机样本上验证：

- RoboCOIN 普通多路 MP4：9 帧稀疏审计通过，episode 保持 A / `joint_video_action`；
- OXE ASU pickle：xyz+rpy 转 canonical rotation vector，34 个窗口中 28 个进入 action，
  6 个与局部 hard interval 相交后只保留 video；其他 OXE 子集没有被错误升级；
- OXE-AugE：按 target robot 拆分，9/9 variant action-eligible，共 96 个 action window；
- RoboMIND `h5_ur_1rgb`：按官方 puppet/puppet contract 读取 HDF5，2 个 action window；
- AgiBot tar 内 `head_color.mp4`：tar member 提取、ffprobe 和稀疏审计通过；因本地缺
  `proprio_stats`，20 个窗口全部保持 video-only；
- Galaxea 实样：四元数分量误报被几何检查消除，从原先 action blocker 恢复为 A；
- Galaxea 该样本生成 24 个 81-step window；普通 abrupt 只做 soft flag，不错误降级 action window。
- InternData-A1 A2D：官方 action schema 优先级和零起始关节名修复后，8 个 action window；
- 六个当前有控制文件的数据集均产出 `81 state / 80 action / 21 video` 联合 case。

自动测试覆盖：

- quaternion `q/-q` 等价；
- extreme action jump 的 hard local interval；
- 连续黑帧和冻结候选；
- action bad interval 只降级重叠窗口；
- OXE pose 派生列和 OXE-AugE next-row target；
- AgiBot observation/proprio tar join 与纳秒时间戳；
- InternData source-key contextual mapping；
- 原有 scan/clean/materialize/windows/cases end-to-end contract。

## 14. 尚未完成的关键工作

1. 为七个数据集建立全局 fingerprint/near-duplicate index，而不是只在单次 clean 内比较。
2. 按 embodiment 配置物理单位、joint/workspace/gripper/velocity/acceleration limits。
3. 完成 OXE 其余子集 action contract；等待 AgiBot 真实 proprio 分片；统计 RoboMIND 未知本体命中率。
4. 增加多相机 PTS drift、跨相机同步和视频-control duration drift 的精确检查。
5. 在人工标定集上评估 VLM success/confidence，再决定是否加入 soft sampling weight。
6. 为 stage 3 接入 dataset-specific success label 和目标域数据平衡规则。

在这些工作完成前，当前管线适合生成可追溯的 Stage 1 语料和已验证 embodiment 的 Stage 2
candidate，不应把 OXE 未审核子集、RoboMIND 未知本体或 observation-only AgiBot 自动打开 action loss。
逐数据集证据、mapping 和 81 帧结果见 `ACTION_DATA_ADMISSION.md`。

## 15. 验证命令

```bash
python3 \
  -m unittest discover -s tests -v

python3 \
  -m compileall -q fastwam_preprocess scripts
```
