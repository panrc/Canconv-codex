# CANNet 一键训练、推理与评估流水线

本仓库新增 `run_pipeline.py`，可一键完成以下流程：

1. **检测/训练模型**：
   * 对 `gf2`、`qb`、`wv3` 三个数据集分别检查是否已存在早停模型 `runs/cannet_<dataset>/weights/best_model.pth`。
   * 若不存在，则自动调用 `canconv/scripts/train.py` 进行训练（使用 `presets.json` 中同名 preset 覆盖数据集路径与参数）。
2. **生成推理结果**：
   * 基于获得的 `best_model.pth`，调用 `canconv/scripts/test.py`，分别在 **Reduced** 与 **OrigScale** 两种分辨率测试集上推理，并保存 `.mat` 结果至 `runs/cannet_<dataset>/weights/best_model/results/`。
3. **可视化**：
   * 使用 `canconv/scripts/show.py` 将 Reduced 结果集中 20 幅样本拼接成大图，保存在 `runs/cannet_<dataset>/vis.png`。
4. **定量评估**：
   * 调用 `evaluate_pansharpening.py` 计算 ERGAS、SAM、Q_avg、HQNR 等指标，并将完整日志输出到 `runs/cannet_<dataset>/eval_results.txt`。
5. **结果汇总**：
   * 流水线完成后，会生成 `summary_eval_results.json`，记录每个数据集对应的权重、可视化图与评估日志路径。

## 快速开始

```bash
# 1. 安装依赖（首次运行需要）
pip install -r requirements.txt

# 2. 一键执行全部数据集
python run_pipeline.py

# 3. 只操作指定数据集
python run_pipeline.py --datasets wv3

# 4. 强制重新训练
python run_pipeline.py --datasets wv3,gf2 --retrain
```

脚本会在执行过程中自动记录日志，若某一步骤失败可直接查看对应的 log / txt 文件排错。

> 注意：
> * 请确保 `presets.json` 中的数据路径有效可访问。
> * 默认使用单 GPU（`cuda:0`）。如需指定，请在运行前设置 `CUDA_VISIBLE_DEVICES`。 