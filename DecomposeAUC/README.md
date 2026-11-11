# AUC 公平性分析

在仓库根目录安装依赖。

```powershell
python -m pip install -e ".[auc]"
```

## 训练并导出预测

仓库附带预处理后的 ASSIST2015 数据。其他数据集需先按 pyKT 的流程预处理。
从 `examples/` 运行以下命令，并用终端打印的 `save_dir` 替换示例路径。

```powershell
python wandb_akt_minimax_train.py --dataset_name assist2015 --fold 0 --use_wandb 0
python wandb_predict.py --save_dir "saved_model/本次训练目录" --use_wandb 0
```

`akt_minimax` 使用训练折的历史前缀校准 2PL，以验证折 EAP 能力分数的三分位切点固定分组。训练按序列组批，在全部有效预测位置保留 AKT 的 BCE，并在已分组学生的历史前缀之后叠加九个有方向的组对排序损失。`--minimax_weight` 控制排序损失权重，默认 1.0。每个训练批次按组和标签最多采样 64 个排序分数。验证按每层最多 256 个固定分位分数近似组对排序损失，按估计的最差组对损失选择检查点。`--batch_size` 控制每步的序列数，可用 `--init_checkpoint` 从同一架构与训练折的普通 AKT 权重初始化。

`akt_minimax` 默认批量组装序列和计算组对损失，并在一次前向中复用 AKT 各层的因果掩码和位置距离。CPU 上将长序列的查询分块，跳过各块之后的未来位置。训练命令与检查点格式沿用原来的接口。每组、每种标签仍均匀无放回地采样，固定种子下的排序抽样和 CPU 分块的 dropout 随机序列会随实现改变。

`wandb_akt_dro_train.py` 提供旧实验中的组感知配对排序对照，可用同一个预测入口导出结果。`seedwandb/akt_minimax.yaml` 是 Minimax 的 sweep 配置。

## 按历史 IRT 三分位分解

回到仓库根目录。将预测路径替换为上述命令生成的非滑窗预测文件。输出目录必须是新目录。

```powershell
python DecomposeAUC/run_posthoc_auc_irt_tertiles.py `
  --train-csv data/assist2015/train_valid.csv `
  --test-csv data/assist2015/test_sequences.csv `
  --item-col concepts --prefix-length 15 --reference-fold 0 `
  --prediction minimax "examples/saved_model/本次训练目录/qid_test_predictions.txt" `
  --output-dir DecomposeAUC/auc_figures/assist2015_irt
```

可重复传入 `--prediction 名称 路径`，让多个模型使用同一组别和评估样本。省略该参数时仅生成分组。测试组别沿用参考折切点，AUC 使用历史前缀之后的作答。矩阵行表示正样本所在组，列表示负样本所在组。对角线是组内 AUC，其余单元格是有方向的 xAUC，按正负样本配对数加权可重建 Global AUC。


## 分解 BD06 的题目级滑窗 AUC

运行 `wandb_predict.py` 后，记录 `Final reported metric` 中的 `windowauclate_mean`，并保留同一运行的 `qid_test_question_window_predictions.txt`。在仓库根目录先生成历史 IRT 分组，再分解题目级滑窗预测。

```powershell
$dataDir = "data/bridge2algebra2006"
$groups = "DecomposeAUC/auc_figures/bd06_irt_groups"
$pred = "examples/saved_model/本次训练目录/qid_test_question_window_predictions.txt"
$reportedAuc = Read-Host "windowauclate_mean"

python DecomposeAUC/run_posthoc_auc_irt_tertiles.py `
  --train-csv "$dataDir/train_valid.csv" --test-csv "$dataDir/test_sequences.csv" `
  --item-col concepts --prefix-length 15 --reference-fold 0 --output-dir $groups

python -m DecomposeAUC.run_question_window_auc `
  --pred-txt $pred --test-csv "$dataDir/test.csv" `
  --cluster-csv "$groups/student_clusters_irt_tertiles.csv" `
  --expected-auc $reportedAuc --output-dir DecomposeAUC/auc_figures/bd06_akt_window_auc
```

使用同一预处理批次的原始 `test.csv` 核对题目、学生和标签。分解使用全部题目级滑窗预测，未获得 IRT 能力标签的学生归入 `unassigned`（未分组）。脚本核对直接计算的 AUC 是否与报告值一致，并核对组对加权重建误差。输出包括热力图、预测分数箱型图、组对权重、人数和样本计数。


## AKT 公平性方法

### MinimaxFairAUC 与分组误差稳健 AUC

- akt_minimaxfairauc 使用独立的 wandb_akt_minimaxfairauc_train.py 入口。按历史 IRT 的低、中、高分组和正负标签做分层目标抽样，沿用 AKT 的历史上下文和原始 logits。训练按论文 Algorithm 1 对模型做 SGD、对组对权重做指数更新，验证分块精确计算最差组对的排序损失。固定步长与每步更新对应论文的更新式。训练预算按目标作答数计算。已有的 akt_minimax 联合 BCE 版本仍可从原入口使用。参考 [论文](https://arxiv.org/abs/2208.10451) 和 [作者代码](https://github.com/zhenhuan-yang/MinimaxFairAUC/tree/bb6d8e0)。
- akt_noisyauc 使用 wandb_akt_noisyauc_train.py。noisy_gamma 接受一个共同的 TV 半径，或按正组 low、mid、high 为行、负组 low、mid、high 为列排列的九个半径。应从独立的组标签可靠性数据估计半径，或把给定半径作为敏感性分析参数。论文的 CLIP 噪声估计针对图像，本项目的 IRT 标签暂无可用的独立校准估计。训练采用分层目标采样、SAM 和投影对偶更新。给定采样批次，直接求解公式 9 的 TV 内层线性最优化，替代作者代码对分布的逐步投影上升。验证用确定性分层样本估计约束，优先保存估计可行的 checkpoint。若所有候选均不可行，保留估计违反量最低的 checkpoint。参考 [论文](https://proceedings.mlr.press/v267/wu25q.html) 和 [作者代码](https://github.com/Purdue-M2/AUC_Fairness_with_Noisy_Groups/tree/6974d23)。

两种模型保存的 checkpoint 都可用 wandb_predict.py 导出预测。

### InterFairAUC

从 examples 运行 wandb_akt_interfairauc_train.py。模型名为 akt_interfairauc，预测复用 wandb_predict.py。训练使用 Vogel 等人的 sigmoid 排序代理和不完全 U 统计量，并对低、中、高三组的每个无序组对加入双向 xAUC 代理差的绝对值。三组求和是对论文两组目标的扩展。实现使用论文的绝对值目标，作者 TensorFlow 实验代码用自适应有符号系数近似这一目标。interfair_weight 控制惩罚权重，interfair_num_pairs 控制每项训练估计的配对采样量。

参考 [论文](https://proceedings.mlr.press/v130/vogel21a.html) 和 [作者代码](https://github.com/RobinVogel/Learning-Fair-Scoring-Functions/tree/bcdaa19)。

### PG-DRO 和 BeShift

在 `examples/` 下运行。两个入口的默认 AKT 参数来自 BD06 fold 0 的超参表。

```powershell
python wandb_akt_pgdro_train.py --dataset_name bridge2algebra2006 --fold 0 --use_wandb 0
python wandb_akt_beshift_train.py --dataset_name bridge2algebra2006 --fold 0 --use_wandb 0
python wandb_predict.py --save_dir "saved_model/本次训练目录" --bz 64 --use_wandb 0
```

- `akt_pgdro` 按训练学生的整段答题正确率取最高 20% 和最低 50%，将低组下采样到两组等大。按公式 5 计算知识点权重。每步均匀抽一个组，以该组批次的知识点平均 BCE 计算公式 4 的加权损失。先指数更新组权重，再以更新后的组权重做 SGD。批次未观测到的知识点本步贡献为零。只在一组出现的知识点权重置零，两组共有知识点的差距全为零时用均匀权重。组权重从均匀分布开始，步长由 `--pgdro_group_lr` 控制，默认 0.01。 [PG-DRO 论文](https://doi.org/10.1109/BigData66926.2025.11401863)
- `akt_beshift` 默认从训练折抽取 30% 的学生作为合成来源，合成条数默认等于来源学生数，可用 `--beshift_num_sequences` 指定。按论文做宽度 0.1 的准确率分箱、指数分配和最大余数取整，再按公式 3–6 计算遗忘、疲劳、知识缺口三种翻转概率。固定折实验用训练折正确率较低的一半估计箱数，可用 `--beshift_num_bins` 指定。论文未给出的衰减系数默认 1.0，可用 `--beshift_decay` 调整。随机选取可达到该箱的来源和行为，优先使用尚未选过的来源，并用条件采样保证正确率落箱。多知识点题目取各知识点翻转概率的均值，并统一翻转同一次作答的知识点副本。合成序列只加入训练折。 [BeShift 论文](https://doi.org/10.1016/j.knosys.2026.115947)

两项均沿用 pyKT 的既有折划分、验证 AUC 选 checkpoint 和普通 AKT 推理。BD06 测试仍按历史 IRT 三分位分解 `windowauclate_mean`。论文的 EdNet/EEDI 实验筛选、PG-DRO 的 ABROCA 和 BeShift 的 FairAUC 属于原论文实验设置。

### xOrder 和 FairPOT

这两个方法使用同一个普通 AKT checkpoint 的题目级滑窗预测。在 `examples/` 下运行测试预测，再用 `--calibration-dir` 导出该 checkpoint 的验证折预测和原始学生序列。导出使用与测试相同的 pyKT 滑窗和 late-mean 聚合。

```powershell
python wandb_predict.py --save_dir "saved_model/本次 AKT 训练目录" --bz 64 --use_wandb 0
python wandb_predict.py --save_dir "saved_model/本次 AKT 训练目录" --bz 64 --use_wandb 0 `
  --calibration-dir "../DecomposeAUC/auc_figures/bd06_akt_calibration"
```

按上面的 BD06 分解流程生成 IRT 分组，`--reference-fold` 填 checkpoint 的验证折。生成的 `reference_scores.csv` 是拟合集的分组，`student_clusters_irt_tertiles.csv` 是测试集的分组，两者使用同一套 IRT 参数和切点。运行题目级滑窗分解时，`--expected-auc` 填测试集报告的原始 `windowauclate_mean`，并附加以下参数。

| 参数 | 内容 |
| --- | --- |
| --calibration-pred-txt | `--calibration-dir` 下的 `qid_validation_question_window_predictions.txt` |
| --calibration-test-csv | `--calibration-dir` 下的 `calibration.csv` |
| --calibration-cluster-csv | IRT 分组输出中的 `reference_scores.csv` |
| --postprocessor akt_xorder --xorder-penalty 权重 | 采用作者的动态规划和按秩插值。默认按 high、low、mid 顺序执行论文的迭代合并。可用 --xorder-groups 显式指定顺序。每次合并需要二次方计算和存储。--xorder-calibration-limit 可选，用于明确指定每侧的确定性秩抽样上限。 |
| --postprocessor akt_fairpot --transport-fraction 比例 | 采用论文的全局 AUC 版本，按一维最优传输调整目标组分数的上尾。默认将 low 组向 high 组映射，可用 --target-group 和 --reference-group 指定。参数在拟合集上选定。 |
| --postprocessor akt_fairpot --fairpot-max-auc-drop 容差 | 在验证折扫描比例 0、0.1、…、1.0，输出 `fairpot_validation_sweep.csv` 和 Pareto 标记。在全量 Global AUC 下降不超过容差的候选中选双向 xAUC 差最小的一项，依次用更高的 AUC 和更小的比例打破并列。用选中的比例调整测试预测。 |

拟合集和评估集的学生 ID 必须来自同一 ID 空间。输出的 global_auc 和 reconstructed_auc 使用调整后的分数，baseline_auc 记录调整前的 AUC。unassigned 保留在全量矩阵中。

xOrder 参考 [论文](https://arxiv.org/abs/2307.14668) 和 [作者代码](https://github.com/cuis15/xorder/tree/6dde5a1)。FairPOT 的一维传输使用单调耦合计算均匀质量的重心，对相同输入分数的重心取均值后做线性插值。比例扫描与 Pareto 判定沿用论文，选择规则用验证折 AUC 容差适配本实验。 [FairPOT 论文公式 1–4 与算法 1](https://arxiv.org/html/2508.03940v1)

## 历史分析入口

| 脚本 | 用途 |
| --- | --- |
| `trans.py` | 用测试序列 CSV 将 pyKT 预测 TXT 展开为含学生 ID 的交互长表。 |
| `cluster_students.py` | 用 2PL/3PL 能力和 person-fit 指标生成分组。 |
| `run_posthoc_auc_irtwofit.py` | 按固定能力切点生成四组。 |
| `run_posthoc_auc_irt_localband.py` | 用本地 Bookmark 风格切点生成三组。 |
| `plot_auc_matrix.py` | 用交互长表与学生分组生成 AUC、配对权重、交互计数和热力图。 |
| `plot_prediction_scores_by_cluster.py` | 按组和标签绘制预测分数箱型图。 |
| `metrics/compute_auc_metrics.py` | 复算历史序列级 AUC 和矩阵均值。单标签序列沿用旧脚本的人工补标签规则。 |

每个入口支持 `--help`。历史固定四组和本地三组方法在测试作答上拟合 IRT，适用于事后诊断。分析时可用 `--pred-txt`、`--seq-csv`、`--cluster-csv` 等参数显式指定本次输入路径。
