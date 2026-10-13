"""text_cleaner 单元测试：清洗、分句、分词。"""
import pytest

from src.preprocessing.text_cleaner import clean_text, split_sentences, tokenize


def test_clean_text_removes_html_and_noise():
    raw = "<p>惠州 发展 ！！！人工智能</p>"
    out = clean_text(raw)
    assert "<" not in out and ">" not in out
    assert "！！" not in out  # 重复标点被折叠
    assert "人工智能" in out


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_split_sentences():
    text = "惠州发展电子信息产业。打造产业高地！你怎么看？"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0].endswith("。")


def test_split_sentences_filters_short():
    assert split_sentences("好。") == []  # 过短碎片被过滤


def test_tokenize_removes_stopwords_and_short():
    tokens = tokenize("惠州的发展和人工智能的未来")
    assert "的" not in tokens  # 停用词
    assert "和" not in tokens
    assert "人工智能" in tokens
    assert all(len(t) >= 2 for t in tokens)  # 单字被过滤


def test_tokenize_digits_removed():
    tokens = tokenize("十五五规划2026年")
    assert not any(t.isdigit() for t in tokens)
