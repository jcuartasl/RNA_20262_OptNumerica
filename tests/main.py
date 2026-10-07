
import numpy as np
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizers.PSO import PSO

dim = 3

sch_lim_inf = np.array([-500]*dim)
sch_lim_sup = np.array([500]*dim)

rastr_lim_inf = np.array([-5.12]*dim)
rastr_lim_sup = np.array([5.12]*dim)

ros_lim_inf = np.array([-5]*dim)
ros_lim_sup = np.array([10]*dim)


def rosenbrock(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def schewfel(x):
    x = np.array(x)
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def rastrigin(x):
    x = np.array(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

opt_PSO = PSO(dim=dim, n_particles=30, function=rosenbrock, upper_bounds=ros_lim_sup, lower_bounds=ros_lim_inf, max_iter=10000)

for _ in range(30):
    opt_PSO.run()

print("\nResultados")
print(opt_PSO.obtener_resultados())