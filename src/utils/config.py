"""配置加载工具：读取 configs/config.yaml，返回 dict，并提供路径解析。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """项目根目录（本文件向上三级：src/utils/config.py -> src -> root）。"""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置。默认读取项目根目录下的 configs/config.yaml。"""
    cfg_path = Path(path) if path else project_root() / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """将 configs.paths 中的相对路径解析为绝对路径。"""
    rel = cfg["paths"][key]
    return project_root() / rel


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """确保 processed / reports / figures 目录存在。"""
    for key in ("processed", "reports", "figures"):
        os.makedirs(resolve_path(cfg, key), exist_ok=True)
