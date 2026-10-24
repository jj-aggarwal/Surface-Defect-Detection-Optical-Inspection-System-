"""Configuration for the inspection pipeline (dataclass + YAML loading)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


@dataclass
class InspectionConfig:
    # --- optics / metrology -------------------------------------------------
    pixel_size_mm: float = 0.05          # mm per pixel at the part surface

    # --- pre-processing -----------------------------------------------------
    flat_field_order: int = 2            # polynomial order of illumination model
    fine_sigma: float = 1.2              # smoothing for small defects (pits)
    coarse_sigma: float = 5.0            # smoothing for large faint defects (stains)
    ridge_sigmas: Tuple[float, ...] = (1.2, 2.0)  # scales for line detector

    # --- detection ----------------------------------------------------------
    ridge_weight: float = 0.55           # ridge z-scores have heavier noise tails; down-weight
    low_threshold: float = 4.0           # robust z-score: region growing
    high_threshold: float = 6.5          # robust z-score: seed required per region
    peak_fraction: float = 0.4           # tighten sharp regions to >= this fraction of their peak
    stain_max_sharp: float = 18.0        # a soft stain has no strong sharp-scale response
    min_area_px: int = 10
    morph_close_px: int = 3
    border_margin_px: int = 6            # ignore image border (edge effects)

    # --- classification -----------------------------------------------------
    scratch_min_aspect: float = 3.5
    crack_min_tortuosity: float = 1.12   # path length / chord of the skeleton
    crack_min_wiggle: float = 0.03       # lateral std of skeleton / chord
    stain_min_area_px: int = 300

    # --- accept / reject criteria (mm) --------------------------------------
    # longest dimension (mm) above which a defect of this class is rejectable.
    max_size_mm: Dict[str, float] = field(default_factory=lambda: {
        "scratch": 3.0, "crack": 0.0, "pit": 0.5, "stain": 2.0,
    })
    max_defect_count: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InspectionConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        if "ridge_sigmas" in data:
            data["ridge_sigmas"] = tuple(data["ridge_sigmas"])
        base = cls()
        for k, v in data.items():
            if not hasattr(base, k):
                raise KeyError(f"Unknown config key: {k}")
            setattr(base, k, v)
        return base

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        d = asdict(self)
        d["ridge_sigmas"] = list(d["ridge_sigmas"])
        text = yaml.safe_dump(d, sort_keys=False)
        if path:
            Path(path).write_text(text)
        return text
