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

    # Disruption model (new in V1)
    inc_c0: float = -2.2  # baseline (lower = fewer incidents)
    inc_c_class: float = 0.03  # effect of class size
    inc_c_B: float = 1.0  # effect of student behavior propensity

    base_loss: float = 0.03  # fraction of day lost per incident
    max_loss: float = 0.35  # cap on time lost per class per day

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_population(p: Params):
    # Create a repeatable random-number generator;
    # using p.seed means the simulation runs the same way every time for the same settings.
    rng = np.random.default_rng(p.seed)

    # Student mastery starts low-to-mid, beta distributions are always between 0 and 1
    # Mean is a / (a + b) or 2 / (2+3) = 2/5 = .4, with a "normalish" distribution
    # variance = (a*b) / ((a+b)^2 * (a + b + 1)) and std = sqrt(variance)
    K = rng.beta(2.0, 3.0, p.n_students)

    # Draw each teacher’s skill from a normal-ish distribution with mean .8
    # "skill" has an array or teacher skills (0, 1]
    skill = rng.beta(16, 4, p.n_teachers)  # mean = 16/(20)=0.80, fairly tight
    # tiny clip to ensure no teacher is exactly zero
    eps = 1e-6
    skill = np.clip(skill, eps, 1-eps)

    # mean 2 / 5 = .4
    B = rng.beta(2.0, 3.0, p.n_students)  # Student behavior propensity in [0,1]
    return rng, K, B, skill

def make_classes(rng, n_students: int, class_size_cap: int):
    # random shuffle of the student indices.
    # If n_students = 5, something like: array([3, 0, 4, 1, 2])
    order = rng.permutation(n_students)

    # "classes" becomes a list of arrays; each array holds the student IDs assigned
    # to one class for today (up to class_size_cap students per class).
    classes = []
    for i in range(0, n_students, class_size_cap):
        classes.append(order[i:i+class_size_cap])
    print("classes = ", classes)
    return classes

def run(p: Params):
    rng, K, B, skill = init_population(p)

    history = []  # (day, K_mean, K_p10, K_p50, K_p90)

    for day in range(1, p.n_days + 1):
        incidents_total = 0
        time_lost_total = 0.0
        classes = make_classes(rng, p.n_students, p.class_size_cap)

        for c_idx, cls in enumerate(classes):
            t = c_idx % p.n_teachers
            class_size = len(cls)

            attention_per_student = p.teacher_time_budget / class_size

            # incidents create time loss for the whole class
            # Probability a student causes an incident today
            x = p.inc_c0 + p.inc_c_class * class_size + p.inc_c_B * B[cls]
            p_inc = sigmoid(x)

            incidents = rng.random(class_size) < p_inc
            inc_count = int(incidents.sum())

            time_loss = min(p.max_loss, p.base_loss * inc_count)
            time_factor = 1.0 - time_loss  # remaining usable instruction time

            incidents_total += inc_count
            time_lost_total += time_loss * class_size  # student-weighted

            # Learning is reduced when time is lost to disruption
            dK = (p.alpha * (attention_per_student * time_factor) * skill[t] * (1.0 - K[cls])
                  - p.forgetting * K[cls])

            K[cls] = np.clip(K[cls] + dK, 0.0, 1.0)

        history.append((
            day,
            float(K.mean()),
            float(np.quantile(K, 0.10)),
            float(np.quantile(K, 0.50)),
            float(np.quantile(K, 0.90)),
            int(incidents_total),
            float(time_lost_total / p.n_students),
        ))

    return history


if __name__ == "__main__":
    p = Params()
    hist = run(p)

    last = hist[-1]
    print("Done.")
    print(f"Day {last[0]} | K_mean={last[1]:.3f} | K_p10={last[2]:.3f} | K_p50={last[3]:.3f} | K_p90={last[4]:.3f}")







