"""Datasets."""

from ._base import DataModule
from .figure_ground_convexity import FigureGroundConvexityDataModule
from .figure_ground_surroundedness import FigureGroundSurroundednessDataModule
from .figure_ground_symmetry import FigureGroundSymmetryDataModule
from .msra10k import MSRA10KDataModule, PrecomputedMSRA10KDataModule

__all__ = [
    "DataModule",
    "FigureGroundConvexityDataModule",
    "FigureGroundSurroundednessDataModule",
    "FigureGroundSymmetryDataModule",
    "MSRA10KDataModule",
    "PrecomputedMSRA10KDataModule",
]
