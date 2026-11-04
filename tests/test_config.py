"""Tests for plumescout.config."""

from pathlib import Path

import pytest

import plumescout
from plumescout import config


def test_package_imports() -> None:
    assert plumescout.__version__


def test_repo_root_contains_configs() -> None:
    assert (config.REPO_ROOT / "pyproject.toml").is_file()
    assert config.CONFIG_DIR.is_dir()


@pytest.mark.parametrize("name", ["sites.yaml", "train.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = config.load_yaml(name)
    assert isinstance(cfg, dict) and cfg


def test_sites_config_enforces_hard_rules() -> None:
    cfg = config.load_yaml("sites.yaml")
    assert cfg["sentinel2"]["collection"] == "COPERNICUS/S2_HARMONIZED"
    assert cfg["emit"]["version"] == "002"
    assert all(aoi["role"] == "test" for aoi in cfg["aois"].values())


def test_train_config_has_no_colour_augmentation() -> None:
    aug = config.load_yaml("train.yaml")["augment"]
    assert set(aug) <= {"hflip", "vflip", "rot90"}


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        config.load_yaml(p)


def test_env_loading_does_not_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("FIRMS_MAP_KEY=from_file\nHF_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FIRMS_MAP_KEY", "from_env")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert config.load_env(env) is True
    import os

    assert os.environ["FIRMS_MAP_KEY"] == "from_env"
    assert os.environ["HF_TOKEN"] == "from_file"


def test_get_secret_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_env", lambda env_path=None: False)
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    with pytest.raises(KeyError, match="GEE_PROJECT"):
        config.get_secret("GEE_PROJECT")
    assert config.get_secret("GEE_PROJECT", required=False) is None


def test_resolve_path_relative() -> None:
    assert config.resolve_path("data/chips") == config.REPO_ROOT / "data" / "chips"
