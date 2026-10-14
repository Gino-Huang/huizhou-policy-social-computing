"""Smoke test: the Streamlit app renders all tabs without exceptions.

Uses streamlit's AppTest (headless). Slow (~10s) but catches KeyErrors,
missing-file crashes, and empty-state failures across every tab.
"""
import pytest
from pathlib import Path

st_test = pytest.importorskip("streamlit.testing.v1", reason="streamlit AppTest unavailable")

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_all_tabs_without_exceptions():
    AppTest = st_test.AppTest
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    errs = list(getattr(at, "exception", []))
    assert not errs, f"app raised: {[str(getattr(e, 'value', e))[:200] for e in errs]}"
    labels = [t.label for t in at.tabs]
    assert labels == ["主题概览", "语义网络", "时间演化", "关注差异", "政策协同网络", "数据导出"]


def test_synergy_tab_uses_real_data_not_fallback():
    AppTest = st_test.AppTest
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    # fallback message must NOT appear when synergy files exist
    infos = [i.value for i in at.info]
    assert not any("尚未生成协同分析结果" in v for v in infos)
    md = " ".join(m.value for m in at.markdown)
    assert "城乡建设" in md  # top synergy node rendered from real data
