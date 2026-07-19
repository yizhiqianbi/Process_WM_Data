# FastWAM 训练路线图

更新日期：2026-07-19

本文只说明工作顺序、阶段门槛和完成定义。已获得的证据见 [Validation Status](reference/VALIDATION_STATUS.md)，具体命令和技术合同由各专题文档维护。

当前范围聚焦 FastWAM，不考虑 Cosmos。LingBot-VLA v2 只作为已经存在的自采数据微调基线，不与 FastWAM 的训练格式或 checkpoint 混用。

## 目标

建立一条可复现的三阶段链路：

1. Stage 1：用大范围 A/B 级视频数据训练 video backbone。
2. Stage 2：用高质量 A 级机器人 action 数据训练带 8/2/1 memory 的 FastWAM。
3. Stage 3：在目标机器人或任务数据上微调并做 held-out/rollout 评测。

统一输入是 `TrainingCaseV1`：81 state、80 action transition、21 video frame、strict 80D canonical 向量和逐维 mask。

## 当前边界

| 模块 | 已有基础 | 仍需完成 |
|---|---|---|
| 数据处理 | 九个逻辑数据源 adapter、A/B/C admission、81/80/21 真实样本回归 | 九库固定 revision 的全量 pipeline、视频验收和 train-only stats |
| Stage 1 | 原七库逐库 smoke、balanced sampler、checkpoint | 将 LingBot-VA/DreamZero 加入配置，九库回归、2-GPU resume 和正式训练 |
| Stage 2 | 8/2/1 memory、因果 mask、原七库 smoke | 九库联合 optimizer/resume、正式 memory pretrain |
| Stage 3 | RoboCOIN 单步微调和 memory inference 基线 | held-out 指标、完整微调和 rollout |
| LingBot-VLA v2 | 自采数据 2000-step 微调与训练集 replay | episode-level held-out、shadow test、closed-loop |

重要限制：FastWAM 当前 `canonical_stage1_all.yaml` 和 `canonical_stage2_memory_all.yaml` 仍只包含原七个数据集。未显式加入 LingBot-VA/DreamZero manifests 前，任何 `all_datasets` 结果都只能称为七库结果。

## P0：冻结可复现基线

正式长训前必须完成：

- 为当前 FastWAM 魔改版本建立自己的 Git fork。
- 审计并提交 memory、canonical loader、sampler、配置和测试。
- 记录 Process_WM_Data、FastWAM、基础权重和数据 revision 的 SHA。
- 确认代码仓库中没有数据、权重、token、`work/` 或训练输出。
- 用固定 commit 重新跑单卡 smoke、checkpoint 和 resume。

当前 FastWAM 工作树若仍是未提交修改，禁止直接启动数天训练，否则无法可靠复现 checkpoint。

**退出门槛**：任意新机器能从固定 commit 和外部路径恢复同一个 smoke run。

## P1：冻结九库数据版本

执行顺序：

1. 按 [Download](data/DOWNLOAD.md) 锁定每库 revision 并生成下载 receipt。
2. 运行 `status` 和 remote/local size `verify`。
3. 先运行结构和引用校验，再对候选数据做稀疏/完整视频解码。
4. 运行九库完整 `scan -> clean -> materialize -> windows -> cases`。
5. 校验所有 `TrainingCaseV1` 引用、shape、mask、split 和 provenance。
6. 只用 `train + A + joint_video_action` case 构建 normalization stats。
7. 保存 manifest、stats 和配置文件的 SHA256。

Action 进入 A 级的依据见 [Action Admission](data/ACTION_ADMISSION.md)，清洗和窗口规则见 [Preprocessing](data/PREPROCESSING.md)。

**退出门槛**：九库均有机器可读的下载报告、case manifest、拒绝原因统计、相机角色统计和 train-only normalization stats；B 级样本不进入 action stats。

## P2：完成九库训练配置

数据冻结后：

- 将 LingBot-VA 和 DreamZero case manifests 加入 Stage 1/2 data config。
- 为每个数据源设置 normalization domain、采样权重和 A/B loss policy。
- 预热 `tar://` 视频 member cache。
- 为全部唯一 prompt 生成并校验 UMT5 embedding cache。
- 导出 resolved config，确认实际 mixture 是九库而非七库。
- 对每库至少解码一个非首窗口，覆盖有效 memory 历史。

**退出门槛**：resolved config、sampler report 和 text/cache manifest 都明确列出九库；缺失文件在训练启动前失败，而不是在某个 rank 中途失败。

## P3：Stage 1 Video Backbone

训练目标：使用 A/B 级视觉窗口训练 video backbone；action/state loss 关闭。

依次执行：

1. 九库逐库单卡 optimizer smoke。
2. 九库混合单卡 smoke。
3. checkpoint 保存和同阶段 resume。
4. 2-GPU balanced sampler、梯度累积和 resume 回归。
5. 再启动多 GPU 正式训练。

验收条件：

- 每库真实视频均成功解码，输入 shape 和相机布局稳定。
- loss/gradient finite，只有 Stage 1 允许的参数更新。
- A/B 级均可参与 video loss，C 级永不进入 loader。
- checkpoint 可被 Stage 2 strict 初始化。
- 1k/5k/10k 等早期 checkpoint 有独立 validation 结果。

配置中的最大 step 只是预算上限，是否继续由 held-out video loss、生成质量、数据重复率和吞吐决定。

## P4：Stage 2 MemoryFastWAM

Stage 2 必须显式加载选定的 Stage 1 video checkpoint 和 ActionDiT 权重。训练接口以 [FastWAM Data Interface](training/FASTWAM_DATA_INTERFACE.md) 为准，memory 和参数冻结规则见 [FastWAM Three-Stage](training/FASTWAM_THREE_STAGE.md)。

依次执行：

1. memory-active 单库 smoke。
2. 九库混合 optimizer smoke。
3. A/B 级 loss-mask 对照。
4. checkpoint 保存、同阶段 resume 和跨阶段初始化。
5. 2-GPU sampler/resume 回归。
6. 多 GPU 正式 memory pretrain。

验收条件：

- current clip 为 21 帧；history 只来自窗口起点之前，不读取当前或未来帧。
- 非首窗口 8/2/1 memory token 数和有效率符合合同。
- A 级产生 finite video/action loss。
- B 级 `action_loss_mask=false`，action loss 严格为零。
- state/action 无效 canonical 维度不参与 normalization 或 loss。
- checkpoint 包含 video、action、memory 和 proprio 所需模块，并能恢复 optimizer/sampler/RNG 状态。

## P5：Stage 3 目标数据微调

第一条完整闭环优先使用已经验证的数据源，再迁移到目标自采数据。

每个目标数据集必须：

- 从 Stage 2 checkpoint strict 恢复，不重新初始化 memory 合同。
- 按 episode/collection condition 划分 train/validation。
- 仅用 train split 计算 normalization stats。
- 保留 Stage 2 zero-shot、finetuned、no-memory 和 memory-length ablation。
- 报告反归一化后的逐维 action error，而不只报告 diffusion loss。
- 在真机前完成 shadow test、限位/速度/方向检查和低速 rollout。

**退出门槛**：至少一个目标数据集有 held-out action 指标、可复现 checkpoint、开放环回放和 rollout 成功率。

## 实验记录

每次训练使用独立目录：

```text
runs/<stage>/<dataset-or-mixture>/<UTC-date>_<git-sha>/
```

至少保存：

```text
command.txt
resolved_config.yaml
code_commits.json
data_manifest.json
normalization_manifest.json
environment.txt
metrics.jsonl
train.log
checkpoints/
evaluation/
```

`data_manifest.json` 应记录 dataset revision、case manifest SHA256、episode/window 数量、split、normalization domain 和 camera-role 分布。

## 停止条件

出现以下任意情况立即停止训练并修复：

- loss/gradient 出现 NaN 或 Inf。
- action mask 与 active dimensions 不一致。
- B 级样本产生非零 action loss。
- memory 读取当前或未来窗口。
- 某个数据源持续解码失败或 sampler 比例偏离 resolved config。
- checkpoint 不完整、不可恢复或缺少跨阶段模块。
- validation 持续恶化而 training loss 继续下降。

## 完成定义

- [ ] Process_WM_Data 和 FastWAM 均固定到可访问 commit。
- [ ] 九库下载、全量 manifest、视觉检查和 train-only stats 有可复现报告。
- [ ] Stage 1 完成单卡、2-GPU、checkpoint/resume 和正式 validation。
- [ ] Stage 2 完成 memory 因果性、A/B mask、单卡、2-GPU 和 checkpoint/resume。
- [ ] Stage 3 至少一个目标任务完成 held-out、open-loop 和 rollout。
- [ ] 所有结论可由保存的 commit、manifest、配置和评测产物复现。

具体已完成项不要在本文重复维护，统一更新到 [Validation Status](reference/VALIDATION_STATUS.md)。
