# Process WM Data

面向 world-action model 的多数据集下载、清洗、统一语义和模型专用 target preparation 代码。

仓库只保存代码、配置和数据合同，不提交原始数据、生成的 `work/`、模型权重、checkpoint 或 token。原始数据只读，所有清洗 sidecar 和 target overlay 写入独立目录。

## Source 与 Target

支持的 source adapter：

- OXE / Open X-Embodiment、OXE-AugE
- AgiBot-Beta、RoboCOIN、RoboMIND
- Galaxea、InternData-A1
- LingBot-VA RoboTwin/LIBERO data
- DreamZero-DROID data

支持的模型 target：

| Target | 训练格式 | 当前实现 |
|---|---|---|
| FastWAM | 81 state / 80 action / 21 video，80D canonical + mask | 完整 pipeline |
| old LingBot-VA | LeRobot v2 + `action_config` + 30D channel contract + VAE latent | target preparation、latent 提取、完整 checkpoint 训练 wrapper |
| DreamZero | LeRobot v2 + GEAR modality/stats + Hydra profile | target preparation、profile 安装、LoRA/Trainer 训练 wrapper |

> 重要边界：`lingbot_va.py` 和 `dreamzero.py` 是 **source adapter**；`targets/lingbot_va/` 和 `targets/dreamzero/` 才是对应模型的 **target exporter**。旧版 LingBot-VA 与 LingBot-VLA-v2 也不是同一个模型。详见 [Target 输出索引](docs/targets/README.md)。

三种模型可通过同一配置入口 dry-run、训练、续训和查询收据：

```bash
python3 scripts/tune_models.py dry-run \
  --config configs/tuning/take_wrong_item.example.yaml \
  --model fastwam --phase stage3_finetune \
  --output-dir work/tuning/runs/fastwam_stage3 --steps 1
```

完整流程见 [三模型统一微调](docs/training/THREE_MODEL_TUNING.md)。
天机自采数据的 FastWAM 单轨迹想象 vs GT 实验见
[FastWAM 天机单轨迹过拟合](docs/training/FASTWAM_TIANJI_OVERFIT.md)。

## 安装

```bash
git clone git@github.com:yizhiqianbi/Process_WM_Data.git
cd Process_WM_Data

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## FastWAM Pipeline

```bash
export FASTWAM_DATA_ROOT=/path/to/robot_dataset

python3 scripts/run_pipeline.py \
  --datasets robocoin galaxea \
  --max-episodes 20 \
  --verify-files
```

处理阶段：

```text
scan -> clean -> materialize -> windows -> cases
```

核心输出为 `TrainingCaseV1`：20 Hz、81/80/21、80D state/action、逐维 valid mask、五个语义相机 slot，以及 A/B/C admission。

## Old LingBot-VA Target

输入必须是 LeRobot v2。下面以 15D 右臂自采数据为例：

```bash
python3 scripts/prepare_lingbot_va_target.py prepare \
  --source-root /data/take_wrong_item_right_arm_v2 \
  --output-root /data/targets/take_wrong_item_lingbot_va \
  --profile take_wrong_item_right_arm \
  --train-episodes-file /data/splits/train_episodes.txt \
  --verify-files
```

它会把 `[0:7] + [14]` 压缩成 8D source action，并映射到旧版 LingBot-VA 的 right-joint channels `21..27` 和 right-gripper channel `29`。同时生成 `action_config`、q01/q99、model profile 和 VAE latent jobs。

VAE/T5 latent 必须在旧 LingBot-VA 模型环境提取：

```bash
/path/to/lingbot/python scripts/extract_lingbot_va_latents.py \
  --target-root /data/targets/take_wrong_item_lingbot_va \
  --model-root /models/lingbot-va-base \
  --lingbot-repo /code/lingbot-va \
  --device cuda

python3 scripts/prepare_lingbot_va_target.py validate \
  --root /data/targets/take_wrong_item_lingbot_va \
  --require-latents \
  --verify-files
```

## DreamZero Target

```bash
python3 scripts/prepare_dreamzero_target.py prepare \
  --source-root /data/take_wrong_item_right_arm_v2 \
  --output-root /data/targets/take_wrong_item_dreamzero \
  --profile take_wrong_item_right_arm \
  --verify-files
```

它会生成 GEAR `modality.json`、embodiment、absolute/relative stats、语言列和 Hydra profile。
对已有 enum/projector 的 `xdof` 安装数据 profile：

```bash
python3 scripts/install_dreamzero_profile.py \
  --target-root /data/targets/take_wrong_item_dreamzero \
  --dreamzero-repo /code/dreamzero
```

## 非破坏输出

两个新增 target 默认使用 `--link-mode symlink`：

- `meta/` 独立复制和更新。
- 未修改的大体积 `data/`、`videos/` 使用链接。
- 需要 action 压缩或增加语言列时，只重写输出 Parquet。
- 输出先写 staging，再原子 rename；已存在的目标目录不会覆盖。

可改用 `--link-mode hardlink` 或 `--link-mode copy`。

## 文档入口

完整导航见 [Documentation Index](docs/README.md)。主要入口：

1. [Target 输出索引](docs/targets/README.md)
2. [FastWAM Target](docs/targets/FASTWAM.md)
3. [Old LingBot-VA Target](docs/targets/LINGBOT_VA.md)
4. [DreamZero Target](docs/targets/DREAMZERO.md)
5. [统一清洗管线](docs/data/PREPROCESSING.md)
6. [当前验收状态](docs/reference/VALIDATION_STATUS.md)
7. [三模型统一微调](docs/training/THREE_MODEL_TUNING.md)

## 仓库结构

```text
configs/targets/          模型 target profiles 与固定上游 revision
fastwam_preprocess/       source adapter、清洗和 FastWAM canonical 实现
targets/                  FastWAM、old LingBot-VA、DreamZero target 代码
scripts/                  下载、清洗、target preparation 和验证入口
tests/                    单元测试与真实 schema 回归
docs/                     数据、target、训练和状态文档
work/                     本地生成物，不提交 Git
```

## 验证与导出

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q fastwam_preprocess targets tuning scripts tests
scripts/export_code.sh /tmp/Process_WM_Data.tar.gz
```

当前三模型已完成真实单步与断点恢复 smoke；全量下载、九库生产训练、LingBot-VA
全部 latent 和闭环评测仍有独立 readiness gate。详见 [当前验收状态](docs/reference/VALIDATION_STATUS.md)。
