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


def init_population(p: Params):
    # Create a repeatable random-number generator;
    # using p.seed means the simulation runs the same way every time for the same settings.
    rng = np.random.default_rng(p.seed)

    # Student mastery starts low-to-mid
    K = rng.beta(2.0, 3.0, p.n_students)

    # Draw each teacher’s skill from a normal distribution (mean +/- spread) so teachers
    # vary realistically instead of all being identical.  "skill" have array or teacher skills
    skill = rng.normal(p.teacher_skill_mean, p.teacher_skill_sd, p.n_teachers)
    # skill = clip to [0.1, 1.0] so nobody is "zero"
    skill = np.clip(skill, 0.1, 1.0)
    return rng, K, skill

def make_classes(rng, n_students: int, class_size_cap: int):
    # random shuffle of the student indices.
    # If n_students = 5, something like: array([3, 0, 4, 1, 2])
    order = rng.permutation(n_students)
    classes = []
    for i in range(0, n_students, class_size_cap):
        classes.append(order[i:i+class_size_cap])
    print("classes = ", classes)
    return classes

def run(p: Params):
    rng, K, skill = init_population(p)

    history = []  # (day, K_mean, K_p10, K_p50, K_p90)

    for day in range(1, p.n_days + 1):
        classes = make_classes(rng, p.n_students, p.class_size_cap)

        for c_idx, cls in enumerate(classes):
            t = c_idx % p.n_teachers
            class_size = len(cls)

            attention_per_student = p.teacher_time_budget / class_size

            # Simple learning: more attention + more skilled teacher -> more growth
            dK = (p.alpha * attention_per_student * skill[t] * (1.0 - K[cls])
                  - p.forgetting * K[cls])

            K[cls] = np.clip(K[cls] + dK, 0.0, 1.0)

        history.append((
            day,
            float(K.mean()),
            float(np.quantile(K, 0.10)),
            float(np.quantile(K, 0.50)),
            float(np.quantile(K, 0.90)),
        ))

    return history


if __name__ == "__main__":
    p = Params()
    hist = run(p)

    last = hist[-1]
    print("Done.")
    print(f"Day {last[0]} | K_mean={last[1]:.3f} | K_p10={last[2]:.3f} | K_p50={last[3]:.3f} | K_p90={last[4]:.3f}")







