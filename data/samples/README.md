# 示例数据说明

本目录下的数据为**人工编写的合成演示数据**，仅用于让仓库克隆后可直接运行
pipeline 与 Streamlit 仪表盘。所有政策段落、热点标题、热度数值均为虚构，
不代表任何真实政策文本或真实热搜榜单。

- `sample_policy_segments.csv`：24 条政策文本片段，覆盖电子信息、石化能源新材料、
  人工智能、就业、教育医疗、交通、海洋经济、生态环境等"十五五"主题。
- `sample_hotspot_snapshots.csv`：24 条热点快照，平台包括微博 / 抖音 / 小红书 / 百度，
  时间为 2026-10。

**真实数据**请放在 `data/raw/`（已被 `.gitignore` 忽略，不会提交）：
- 政策原文：`.txt` / `.md` / `.pdf`，用 `src/ingestion/policy_loader.py` 读入；
- 热点快照：按 `docs/data_dictionary.md` 的 schema 整理为 CSV，用
  `src/ingestion/hotspot_collector.py` 导入。

注意：合成样本量很小（各 ≤30 行），统计结论仅作流程演示，不具研究效力。
