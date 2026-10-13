"""中文文本清洗：去 HTML、去噪音标点、分句、jieba 分词（含停用词过滤）。"""
from __future__ import annotations

import re
from pathlib import Path

import jieba

# 去 HTML 标签
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# 连续空白（含全角空格）压缩为单个空格
_WS_RE = re.compile(r"[\s\u3000]+")
# 重复标点噪音（如 "！！！！"、"。。。。")
_REP_PUNCT_RE = re.compile(r"([！!？?。，,.、；;：:]{2,})")
# 中文分句：保留句末标点
_SENT_SPLIT_RE = re.compile(r"([^。！？!?…\n]+[。！？!?…]?)")

# 内置最小停用词表（项目可替换为 data/ 下的外部词表）
DEFAULT_STOPWORDS = frozenset(
    "的 了 和 是 在 有 与 及 或 为 对 将 被 把 从 到 以 于 而 及 其 这 那 什么 怎么 我们 你们 他们 她 它 个 种 些 并 但 若 如 如果 因为 所以 由于".split()
)


def load_stopwords(path: str | Path | None = None) -> frozenset[str]:
    """从文件加载停用词（每行一个词）；path 为空时返回内置词表。"""
    if path is None:
        return DEFAULT_STOPWORDS
    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.add(w)
    return frozenset(words)


def clean_text(text: str) -> str:
    """清洗单段文本：去 HTML、压缩空白、折叠重复标点、去首尾空格。"""
    if not text:
        return ""
    text = _HTML_TAG_RE.sub("", text)
    text = _REP_PUNCT_RE.sub(lambda m: m.group(1)[0], text)  # "！！！" -> "！"
    text = _WS_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str, min_len: int = 4) -> list[str]:
    """按中文标点分句；过滤过短的碎片。"""
    text = clean_text(text)
    if not text:
        return []
    sentences = [s.strip() for s in _SENT_SPLIT_RE.findall(text)]
    return [s for s in sentences if len(s) >= min_len]


def tokenize(
    text: str,
    stopwords: frozenset[str] | None = None,
    min_token_len: int = 2,
) -> list[str]:
    """jieba 精确模式分词；过滤停用词、纯数字、过短 token。

    参数:
        min_token_len: 保留的最小词长。中文单字信息量低，默认只保留 >=2 字的词。
    """
    stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
    tokens: list[str] = []
    for tok in jieba.cut(clean_text(text), cut_all=False):
        tok = tok.strip()
        if not tok or tok in stopwords:
            continue
        if tok.isdigit():
            continue
        if len(tok) < min_token_len:
            continue
        tokens.append(tok)
    return tokens


def jieba_tokenizer(text: str) -> list[str]:
    """供 sklearn TfidfVectorizer 调用的模块级分词函数（可被序列化引用）。"""
    return tokenize(text)
