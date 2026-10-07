
import numpy as np
import pandas as pd

class PSO:
    def __init__(self, dim, n_particles, function, upper_bounds, lower_bounds, max_iter=10000, w = 0.7, lr_1 = 1.5 , lr_2=1.5):
        self.dim = dim
        self.n_particles = n_particles
        self.function= function
        self.bounds = (lower_bounds, upper_bounds)
        self.max_iter = max_iter
        self.w = w
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.v_max = 0.5 * (self.bounds[1] - self.bounds[0])
        self.results = []


    #Define una población inicial de partículas dentro de los límites especificados
    def __pob_inicial(self):
        pob = np.random.rand(self.n_particles, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return pob

    #Define las velocidad inicial para cada partícula dentro de los límites especificados
    def __vel_inicial(self):
        vel = np.random.rand(self.n_particles, self.dim) * (self.v_max) - (self.v_max / 2)
        return vel

    #Ejecuta el algoritmo de optimización por enjambre de partículas (PSO) para encontrar la mejor solución a la función objetivo dada.
    def run(self):
        seed = np.random.randint(0, 10000)
        np.random.seed(seed)
        
        pob = self.__pob_inicial()
        vel = self.__vel_inicial()

        pBest = pob.copy()
        pBest_val = np.array([self.function(x) for x in pob])

        gBest = pob[np.argmin(pBest_val)].copy()
        gBest_val = np.min(pBest_val)

        x = pob.copy()

        for _ in range(self.max_iter):
            vel = vel * self.w + self.lr_1 * np.random.rand(*x.shape) * (pBest - x) + self.lr_2 * np.random.rand(*x.shape) * (gBest - x)
            vel = np.clip(vel, -self.v_max, self.v_max)
            x = np.clip(x + vel, self.bounds[0], self.bounds[1])

            f = np.array([self.function(x) for x in x])

            mejora = f < pBest_val
            pBest[mejora] = x[mejora]
            pBest_val[mejora] = f[mejora]

            idx = np.argmin(pBest_val)
            gBest = pBest[idx].copy()
            gBest_val = pBest_val[idx]
            

        self.results.append({
            'gBest': np.round(gBest.copy(), 4),
            'gBest_val': np.round(gBest_val, 4),
            'seed' : seed
        })

        print("Corriendo el algoritmo con semilla: ", seed)
        print("Mejor solución encontrada: ", np.round(gBest.copy(), 4))
        print("Valor de la función objetivo: ", np.round(gBest_val, 4))

    #Retorna los resultados en formato de DataFrame
    def obtener_resultados(self):
        return pd.DataFrame(self.results)





