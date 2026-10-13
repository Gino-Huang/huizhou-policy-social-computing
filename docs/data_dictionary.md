# 数据字典

## 1. 政策文档层

### policy_documents
| 字段 | 类型 | 说明 |
|---|---|---|
| doc_id | string | 文档唯一标识，如 `doc_001` |
| source_path | string | 原始文件路径 |
| n_chars | int | 文档字符数 |

### text_segments
| 字段 | 类型 | 说明 |
|---|---|---|
| segment_id | string | 片段唯一标识，如 `doc_001_s0001` |
| doc_id | string | 所属文档 |
| section | string | 段落-句子编号，如 `P3-S2` |
| text | string | 清洗后的句子文本 |
| order | int | 文档内顺序 |

## 2. 热点快照层

### hotspot_snapshots（快照 CSV schema）
| 字段 | 类型 | 说明 |
|---|---|---|
| snapshot_id | string | 快照唯一标识 |
| platform | string | 平台：微博 / 抖音 / 小红书 / 百度 |
| captured_at | datetime | 抓取时间（`YYYY-MM-DD HH:MM:SS`） |
| title | string | 热点标题 |
| rank | int | 榜单排名（越小越靠前） |
| raw_heat | float | 平台原始热度值（**各平台口径不同，禁止跨平台求和**） |
| url | string | 链接（可为空） |

## 3. 匹配结果层

### matches（语义匹配输出）
| 字段 | 类型 | 说明 |
|---|---|---|
| hotspot_id | string | 热点标识 |
| policy_id | string | 政策片段标识；`None` 表示无可靠匹配 |
| similarity | float | 余弦相似度 [0,1] |
| rank | int | 该热点下的检索排名；无匹配时为 0 |

## 4. 标注层（评估用）

### annotations
| 字段 | 类型 | 说明 |
|---|---|---|
| pair_id | string | `{hotspot_id}__{policy_id}`（双下划线分隔） |
| relevant | int | 0=不相关，1=部分相关，2=高度相关 |
