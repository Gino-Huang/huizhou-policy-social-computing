# 惠州市"十五五"政策语义网络与社会热点分析平台

**Huizhou Policy & Public Attention Lab** · 计算社会科学课程 / 求职作品集项目

用 NLP + 网络分析回答一个问题：**政府"十五五"规划里强调的议题，和老百姓
在热搜上真正关注的议题，是同一回事吗？**

## 研究问题

- **RQ1（主题结构）**：惠州"十五五"政策文本的主题结构是什么？
- **RQ2（语义关联）**：政策主题与社会热点之间存在怎样的语义关联？哪些政策议题在舆论场中有回响，哪些"自说自话"？
- **RQ3（关注差异）**：政策议程与公众注意力在主题分布和时间演化上有何差异？差异是稳定的还是事件驱动的？

## 架构

```
政策原文 (.txt/.md/.pdf)                热点榜单快照 (CSV)
        │                                        │
        ▼                                        ▼
┌─ 预处理 ────────────────────┐   ┌─ 采集校验 ─────────────┐
│ 清洗→分句→jieba分词→去重     │   │ schema校验→时间解析→去重 │
└──────────────┬──────────────┘   └────────────┬───────────┘
               │                               │
               ▼                               ▼
┌─ 主题建模 ──┐                   ┌─ 语义匹配 ───────────────┐
│ TF-IDF基线   │                   │ 句向量余弦 / TF-IDF基线    │
│ KMeans+轮廓系数│                   │ Top-K + 阈值(0.55)→匹配表  │
└──────┬──────┘                   └─────────────┬────────────┘
       │                                        │
       └──────────────┬─────────────────────────┘
                      ▼
        ┌─ 语义网络 ─────────────────┐
        │ 政策-热点二部图（边权=相似度）│
        │ 加权度排名/投影/社区发现     │
        └──────────────┬─────────────┘
                       ▼
        ┌─ 差异与演化 ───────────────┐
        │ 主题分布 JSD 散度           │
        │ 日频计数→滑动平均→z-score爆发│
        │ 分平台热度（禁止跨平台求和） │
        └───────────────────────────┘
```

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
