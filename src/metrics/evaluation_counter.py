class EvaluationCounter:
    """
    Lleva el conteo de evaluaciones realizadas durante una optimización.

    Se registran por separado:
    - Evaluaciones de la función objetivo f.
    - Evaluaciones del gradiente ∇f.

    Para comparar los algoritmos bajo un presupuesto común se utiliza:

        E_eq = N_f + 2*n*N_grad

    donde:
        N_f    = número de evaluaciones de f
        N_grad = número de evaluaciones del gradiente
        n      = dimensión del problema
    """

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("La dimensión debe ser mayor que cero.")

        self.dimension = dimension
        self.f_evaluations = 0
        self.gradient_evaluations = 0

    def count_f(self, amount: int = 1):
        """Registra evaluaciones de la función objetivo."""
        if amount < 0:
            raise ValueError("La cantidad de evaluaciones no puede ser negativa.")

        self.f_evaluations += amount

    def count_gradient(self, amount: int = 1):
        """Registra evaluaciones del gradiente."""
        if amount < 0:
            raise ValueError("La cantidad de evaluaciones no puede ser negativa.")

        self.gradient_evaluations += amount

    @property
    def equivalent_evaluations(self) -> int:
        """
        Calcula el número de evaluaciones equivalentes.

        Se considera que una evaluación del gradiente equivale
        a 2*n evaluaciones de la función objetivo.
        """
        return (
            self.f_evaluations
            + 2 * self.dimension * self.gradient_evaluations
        )

    def reset(self):
        """Reinicia todos los contadores."""
        self.f_evaluations = 0
        self.gradient_evaluations = 0

    def to_dict(self):
        """Devuelve las métricas en formato de diccionario."""
        return {
            "f_evaluations": self.f_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "equivalent_evaluations": self.equivalent_evaluations,
        }