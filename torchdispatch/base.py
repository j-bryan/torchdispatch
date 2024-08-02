from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class ConfidenceInterval:
    mean: float
    std: float
    confint: tuple[float, float]
    alpha: float


@dataclass
class OptimizationResult:
    x: torch.Tensor | dict[str, torch.Tensor] | np.ndarray
    storage_level: torch.Tensor | np.ndarray
    y: torch.Tensor | np.ndarray
    penalty: float
    success: bool
    nfev: int
    niter: int
    message: str = ""
    confint: ConfidenceInterval | None = None
