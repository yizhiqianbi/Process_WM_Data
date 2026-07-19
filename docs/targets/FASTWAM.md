# FastWAM Target

[Target 输出索引](README.md)

FastWAM 是仓库最早且最完整的 target。它消费所有 source adapter 的统一清洗结果，并生成：

```text
scan -> clean -> materialize -> windows -> cases
```

核心合同：

- 20 Hz control timeline。
- 81 state points、80 action transitions、21 video points。
- 80D canonical state/action 与逐维 valid mask。
- 五个固定语义相机 slot。
- A 级 video + action、B 级 video-only、C 级 reject。

执行入口：

```bash
python3 scripts/run_pipeline.py \
  --datasets all \
  --output-root work/stage_pipeline \
  --workers 2 \
  --verify-files
```

输出入口：

```text
work/stage_pipeline/<dataset>/cases/training_cases.jsonl
```

详细数据接口见 [FastWAM Data Interface](../training/FASTWAM_DATA_INTERFACE.md)，三阶段训练见 [FastWAM Three-Stage](../training/FASTWAM_THREE_STAGE.md)。

模型侧的 `TrainingCaseV1` loader、8/2/1 memory、联合推理与固定评测实现以代码 patch 固定在
[`integrations/fastwam/`](../../integrations/fastwam/README.md)，基线 SHA 和应用命令以该目录为准。
