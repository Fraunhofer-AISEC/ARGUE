from dataclasses import dataclass

import numpy as np


@dataclass
class ExperimentData:
    """
    All data needed for the experiment
    """
    train_target: dict
    train_alarm: tuple
    train_attention: np.ndarray
    val_target: dict
    val_alarm: tuple
    val_attention: np.ndarray
    test_target: dict
    test_alarm: tuple
    test_attention: np.ndarray
    data_shape: tuple
    input_shape: tuple


