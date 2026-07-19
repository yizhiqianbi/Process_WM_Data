# FastWAM Multi-Dataset Preprocessing

面向 FastWAM 预训练和微调的多数据集下载、清洗、统一格式与训练接入代码。仓库支持以下九个逻辑数据源：

- OXE / Open X-Embodiment
- OXE-AugE
- AgiBot-Beta
- RoboCOIN
- RoboMIND
- Galaxea
- InternData-A1
- LingBot-VA
- DreamZero-DROID

仓库只保存代码、配置和数据合同，不提交原始数据、生成的 `work/`、模型权重或认证 token。原始数据只读，所有索引、统计和 canonical sidecar 写入 `work/`。

> 当前能力边界：九个数据源均有 adapter 和真实样本回归，但“样本级通过”不等于“九库全量生产训练已完成”。当前验收结果以 [Validation Status](docs/reference/VALIDATION_STATUS.md) 为准，后续执行顺序以 [Roadmap](docs/ROADMAP.md) 为准。

## 统一合同

每个训练窗口统一为：

- 控制时间线：20 Hz，`81` 个 state 点和 `80` 个 action transition，共 4 秒。
- 视频时间线：offset `0, 4, ..., 80`，共 `21` 帧。
- 机器人向量：保留 native 数值，同时映射为 strict `80D` canonical state/action。
- 有效维度：必须读取 `state_dim_valid_mask` 和 `action_dim_valid_mask`；canonical 中的零值不代表该维度存在。
- 相机：按语义角色映射到五个固定 slot，不依赖源目录或字段顺序。
- 数据分级：A 级可用于 video + action，B 级只用于 video，C 级拒绝。
- 切分：以 episode/lineage 为单位，防止相邻窗口或增强副本跨 train/validation 泄漏。

## 快速开始

```bash
git clone git@github.com:yizhiqianbi/Process_WM_Data.git
cd Process_WM_Data

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

export FASTWAM_DATA_ROOT=/path/to/robot_dataset
python3 scripts/run_pipeline.py \
  --datasets robocoin galaxea \
  --max-episodes 20 \
  --verify-files
```

运行全部 adapter：

```bash
python3 scripts/run_pipeline.py \
  --datasets all \
  --workers 2 \
  --verify-files
```

第一轮全量处理不建议启用 `--decode-videos`。先用 manifest 和稀疏视觉检查过滤 C 级数据，再安排完整视频解码。

## 处理阶段

| 阶段 | 输入 | 主要输出 |
|---|---|---|
| `scan` | 原始数据 | `episodes.jsonl`、artifact 索引、源 schema |
| `clean` | scan manifest | A/B/C 质量等级、坏区间、action admission |
| `materialize` | clean manifest | 20 Hz canonical state/action、80D mask、源帧对齐 |
| `windows` | canonical episode | 可用 81/80/21 窗口范围 |
| `cases` | window manifest | `TrainingCaseV1`、loss mask、split、normalization domain |

各阶段可以独立运行，也可以由 `scripts/run_pipeline.py` 串联。数据集专用入口位于 `scripts/preprocess_*.py`。

## 文档入口

完整导航见 [Documentation Index](docs/README.md)。建议按以下顺序阅读：

1. [Validation Status](docs/reference/VALIDATION_STATUS.md)：哪些结果已验证，哪些仍是待办。
2. [Roadmap](docs/ROADMAP.md)：FastWAM 三阶段训练的实际推进顺序和门槛。
3. [Preprocessing](docs/data/PREPROCESSING.md)：清洗分层、坏区间、窗口准入和阈值校准。
4. [Action Admission](docs/data/ACTION_ADMISSION.md)：每个数据源为何能或不能进入 action loss。
5. [FastWAM Data Interface](docs/training/FASTWAM_DATA_INTERFACE.md)：`TrainingCaseV1` 到模型 batch/loss 的合同。
6. [FastWAM Three-Stage Training](docs/training/FASTWAM_THREE_STAGE.md)：Stage 1/2/3、memory 和 checkpoint 交接。

下载、AgiBot、LingBot-VA 与 DreamZero 的专题说明也统一放在 `docs/` 下。

## 仓库结构

```text
configs/                 数据集、清洗和训练 profile
fastwam_preprocess/      adapter、清洗、materialize 与统一合同实现
scripts/                 下载、预处理、统计、校验和训练辅助脚本
tests/                   单元测试和真实 schema 回归
docs/                    文档索引、路线图和专题文档
work/                    本地生成物，不提交 Git
```

## 验证

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q fastwam_preprocess scripts
```

生产训练前还必须完成：固定数据 revision、全量文件校验、train-only normalization stats、真实视频解码抽检，以及 FastWAM checkpoint/resume 回归。具体门槛见 [Roadmap](docs/ROADMAP.md)。
