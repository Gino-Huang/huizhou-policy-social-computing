# 惠州市"十五五"发展规划政策主题识别及协同关系研究

**基于 NLP 与复杂语义网络** · 计算社会科学 / 求职作品集项目

以惠州市"十五五"规划纲要及公开配套政策文件为语料，用 NLP 与复杂网络方法
回答：**规划重点关注哪些发展主题？产业、人才、交通、创新与生态等政策主题
之间存在怎样的联系？哪些领域可能成为跨部门协同的关键节点？**

## 研究问题（原计划）

- **RQ1（主题识别）**：规划重点关注哪些发展主题？（TF-IDF + KMeans 主题建模）
- **RQ2（主题关联）**：产业、人才、交通、创新与生态等政策主题之间存在怎样的
  联系？（关键词共现网络、主题关联网络、社群发现）
- **RQ3（协同关键节点）**：哪些领域可能成为跨部门协同的关键节点？
  （中介中心性分析——仅报告网络结构，不引申为政策建议）

> **扩展模块**：政策—社会热点对照分析（语义匹配、二部图、JSD 关注差异、
> 每日热点快照采集）见 `reports/threshold_calibration.md` 与 RQ1–RQ3（扩展）
> 相关报告；热点快照每日自动采集，属独立数据管线。

## 架构

```
政策原文 (.txt/.md/.pdf，7份官方文件)
        │
        ▼
┌─ 预处理 ────────────────────┐
│ 清洗→段落合并→jieba分词→去重  │
└──────────────┬──────────────┘
               ▼
┌─ 主题识别（RQ1）─────────────┐
│ TF-IDF基线 + KMeans聚类       │
│ 15个政策主题 + 代表性关键词    │
└──────────────┬──────────────┘
               ▼
┌─ 协同网络（RQ2/RQ3）──────────┐
│ 关键词共现网络（边权=同段共现） │
│ 主题关联网络（边权=中心向量余弦）│
│ Louvain/贪心社群发现           │
│ 度中心性/中介中心性→协同关键节点 │
└──────────────────────────────┘
```

**扩展模块**（独立管线，不影响主线）：政策—社会热点语义匹配、
二部图、JSD 关注差异、每日热点快照采集。

详细方法见 [reports/methodology.md](reports/methodology.md)。

## 快速开始

```bash
# 1. 克隆并安装
git clone <your-repo-url>
cd huizhou-policy-social-computing
pip install -r requirements.txt

# 2. 跑通测试（模型延迟加载，测试不需要下载任何模型）
pytest tests/ -q

# 3. 启动交互仪表盘（使用 data/samples/ 合成演示数据）
streamlit run app.py
```

真实数据流程：政策原文放入 `data/raw/` → `src/ingestion/policy_loader.py`；
热点快照 CSV 按 schema 整理 → `src/ingestion/hotspot_collector.py`。
数据字典见 [docs/data_dictionary.md](docs/data_dictionary.md)。

> **Windows 用户注意（模型缓存位置）**：首次运行语义匹配时会自动下载约 420MB 的句向量模型，
> 默认存放在 C 盘用户目录（`%USERPROFILE%\.cache\huggingface`）。如需放到 D 盘，运行前设置环境变量：
> ```bat
> setx HF_HOME "D:\huggingface_cache"
> setx HUGGINGFACE_HUB_CACHE "D:\huggingface_cache"
> ```
> 设置后重新打开终端生效。仓库本身只含代码与小样本数据，不含模型文件，克隆不占多少空间。

## 仓库结构

```
├── app.py                  # Streamlit 仪表盘（5 个页面）
├── configs/config.yaml     # 全局参数：模型、阈值、Top-K、平台、随机种子
├── data/
│   ├── raw/                # 真实数据（gitignored，不提交）
│   ├── processed/          # parquet 中间结果（gitignored）
│   └── samples/            # 合成演示数据（可提交，≤30行/表）
├── src/
│   ├── ingestion/          # 政策加载（txt/md/pdf）、热点快照采集
│   ├── preprocessing/      # 中文清洗、分句、jieba分词、去重
│   ├── modeling/           # TF-IDF主题建模、语义匹配、二部网络
│   ├── analytics/          # 时间演化、主题差异（JSD）
│   ├── evaluation/         # P@K/R@K、阈值灵敏度扫描
│   └── utils/              # 配置加载
├── tests/                  # pytest 单元测试
├── notebooks/              # 探索性分析（gitkeep）
├── reports/                # 方法说明、图表输出
└── docs/                   # 数据字典、数据来源、研究局限
```

## 数据来源与许可

- 政策文本：政府官网公开发布的政策文件，仓库**不存放原文**，复现时自行下载放入 `data/raw/`。
- 热点榜单：各平台公开热搜/热榜快照，仅保存标题、排名、热度等公开字段，遵守平台使用条款。
- `data/samples/` 为人工编写的虚构演示数据，**不得用于任何研究结论**。
- 代码拟采用 MIT License；数据遵循原发布机构版权要求。详见 [docs/data_sources.md](docs/data_sources.md)。

## 研究局限（必读）

1. **热搜 ≠ 民意**：榜单是算法、运营与用户行为共同作用的结果，测量的是"被放大的公共注意力"，不是民意本身。
2. **语义相似 ≠ 因果**：向量相似度高只说明语义接近，不能推出因果。本项目不做任何因果断言。
3. **阈值依赖**：网络结构依赖相似度阈值（默认 0.55）；结论必须附带阈值灵敏度分析。
4. **地域归因边界**：全国性热点可能与本地政策文本语义相近但事件本身非本地事务，需人工核验。

完整版见 [docs/limitations.md](docs/limitations.md)。

## 诚实说明

- 所有模型（jieba、sentence-transformers）均在**本地批量运行**，不调用任何外部 API；
- `SemanticMatcher` 采用延迟加载：import 与实例化不会下载模型，只有第一次 `encode()` 才会加载；
- 仪表盘默认提供 TF-IDF 基线一键运行，句向量为可选项；
- 当前 `data/samples/` 数据为合成样本，所有数字结论仅作流程演示。
