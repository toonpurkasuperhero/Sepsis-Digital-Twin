"""Loads the YAML configuration and resolves project paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    settings: dict
    parameters: dict
    interventions: dict

    @property
    def mimic_root(self) -> Path:
        return _resolve(self.settings["mimic_root"])

    @property
    def output_root(self) -> Path:
        return _resolve(self.settings["output_root"])

    @property
    def version(self) -> str:
        return self.settings["pipeline_version"]

    def param_specs(self, layer: str) -> dict:
        """Parameter specs for 'primary' or 'secondary', skipping wearable-only entries."""
        return {k: v for k, v in self.parameters[layer].items() if v.get("itemids")}

    def all_specs(self) -> dict:
        return {**self.param_specs("primary"), **self.param_specs("secondary")}


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_dir: Path | None = None, **overrides) -> Config:
    config_dir = config_dir or PROJECT_ROOT / "config"
    read = lambda name: yaml.safe_load((config_dir / name).read_text())
    settings = read("settings.yaml")
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return Config(settings, read("parameters.yaml"), read("interventions.yaml"))
