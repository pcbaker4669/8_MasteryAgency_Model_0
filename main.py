from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    # population
    n_students: int = 120
    n_teachers: int = 6

    # policy lever
    class_size_cap: int = 30

    # Teacher resources
    teacher_time_budget: float = 30.0 # attention units per teacher per day

    # Learning dynamics
    alpha: float = 0.05  # Learning rate scale
    forgetting: float = 0.002  # daily forgetting proportional to K

    # Run controls
    n_days: int = 60
    seed: int = 1

    # Teacher skill distribution
    teacher_skill_mean: float = 0.80
    teacher_skill_sd: float = 0.10


def init_popuation(p: Params):
    rng = np.random.default_rng(p.seed)





