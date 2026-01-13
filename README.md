# 基于混合策略与文本融合的增强型 SASRec 推荐系统

本仓库是 [WEB搜索与推荐系统导论] 课程设计的代码实现。本项目针对传统 SASRec 模型存在的**冷启动/短序列推荐难**、**侧面信息利用不足**以及**鲁棒性欠佳**等问题，提出了一套综合解决方案。

## ✨ 主要特性

1.  **多模态特征融合 (Text Fusion)**:
    - 引入物品的文本 Embedding（如 BERT 提取）。
    - 通过 Deep Fusion Layer 将语义信息与 ID Embedding 深度融合。
2.  **对比学习增强 (Contrastive Learning)**:
    - 引入 InfoNCE Loss 作为辅助任务。
    - 使用序列增强（Mask/Crop）最大化同一序列不同视图间的互信息，提升鲁棒性。
3.  **长短序列混合推理 (Hybrid Strategy)**:
    - **短序列 (< Threshold)**: 自动切换使用 N-Gram (1-Gram/2-Gram) 统计模型，捕捉强关联规则。
    - **长序列 (>= Threshold)**: 使用增强型 SASRec 模型，捕捉长距离依赖。

## 📂 目录结构

- `model.py`: **核心模型**。包含 `SASRec` 类，集成了 Text Fusion 模块和 Contrastive Learning Loss 计算。
- `train.py`: **训练脚本**。支持数据加载、模型初始化、联合 Loss 优化 (`Rec_Loss + lambda * CL_Loss`)。
- `evaluate_model.py`: **评估脚本**。实现了 Hybrid Inference 逻辑，根据序列长度动态切换 N-Gram 与 SASRec。
- `analyze_data.py`: **数据分析**。生成数据分布图（序列长度、物品流行度等）。
- `build_ngram.py`: 构建 N-Gram 统计模型。
- `extract_text_features.py`: 提取物品文本特征（预处理）。
- `dataset.py`: PyTorch Dataset 定义。

## 🚀 快速开始

### 1. 环境准备
```bash
pip install torch pandas numpy scikit-learn tqdm matplotlib
```

### 2. 数据准备
请确保数据文件（如 `train_augmented.csv`, `test2.csv`）和预训练 Embedding (`item_embeddings.pkl`) 位于项目根目录。
*(注：由于数据文件较大，未包含在 git 仓库中)*

### 3. 训练模型
```bash
python train.py
```
训练日志将保存在 `train_log_enhanced.txt`。

### 4. 评估模型
```bash
python evaluate_model.py
```
该脚本将加载最佳模型 `sasrec_best.pth` 和 N-Gram 模型 `ngram_model_enhanced.pkl` 进行混合推理评估。

### 5. 数据分析
```bash
python analyze_data.py
```
生成的分布图将保存在 `analysis_output/` 目录。

## 📊 实验结果

| 模型配置 | MRR@10 | 相对提升 (vs Baseline) |
| :--- | :---: | :---: |
| Standard SASRec | 0.0636 | - |
| SASRec + Text Fusion | 0.0662 | +4.1% |
| SASRec + CL | 0.0641 | +0.8% |
| **Hybrid (Ours)** | **0.0673** | **+5.8%** |

## 🔗 参考

- Kang, W. C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation. ICDM.
- Xie, X., et al. (2022). Contrastive Learning for Sequential Recommendation. ICDE.
