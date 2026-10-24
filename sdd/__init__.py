"""Industrial Surface Defect Detection & Optical Inspection System."""
from .config import InspectionConfig
from .pipeline import Inspector, InspectionResult, Defect

__all__ = ["InspectionConfig", "Inspector", "InspectionResult", "Defect"]
__version__ = "1.0.0"
