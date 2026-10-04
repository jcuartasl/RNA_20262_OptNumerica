# Bitácora de Cacería de Alucinaciones (AI Hallucination Hunt)

Este documento registra los hallazgos de errores, alucinaciones o imprecisiones generadas por herramientas de Inteligencia Artificial durante el desarrollo del trabajo, en cumplimiento con el enunciado oficial y la Guía Operativa.

---

## Hallazgo H-01
- **Autor que lo detectó**: Persona 1 (Funciones + Gradient Descent)
- **Fecha**: 2026-10-03
- **Categoría**: Matemáticas
- **Prompt literal**: 
  > *"Dame el gradiente analítico en Python para la función de Rastrigin multidimensional: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))"*
- **Parte relevante de la respuesta de la IA**:
  ```python
  # Respuesta errónea generada por IA:
  grad = 2 * x - 20 * np.pi * np.sin(2 * np.pi * x)
  ```
- **Por qué sospechamos**:
  Al observar la ecuación, el término es $-10 \cos(2\pi x_i)$. La derivada de $\cos(u)$ respecto a $u$ es $-\sin(u)$, por lo que la derivada de $-\cos(u)$ debe ser $+\sin(u)$. La IA colocó un signo menos exterior manteniendo $-20\pi \sin(2\pi x_i)$.
- **Evidencia que demuestra el error**:
  Aplicando la regla de la cadena analítica:
  $$\frac{d}{dx_i} \left[ -10 \cos(2\pi x_i) \right] = -10 \cdot \left( -\sin(2\pi x_i) \cdot 2\pi \right) = +20\pi \sin(2\pi x_i)$$
  Experimento numérico con diferencias finitas centrales ($h = 10^{-6}$):
  En $x = 0.25$:
  - Derivada numérica real: $2(0.25) + 20\pi \sin(\pi/2) = 0.5 + 20\pi(1) \approx 63.3318$
  - Fórmula errónea de la IA: $2(0.25) - 20\pi(1) = 0.5 - 62.8318 \approx -62.3318$
  Discrepancia superior a $125.66$, apuntando en dirección diametralmente opuesta a la pendiente.
- **Corrección**:
  ```python
  grad = 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)
  ```
- **Lección aprendida**:
  Los LLMs son propensos a equivocaciones de signos en derivadas trigonométricas encadenadas con coeficientes negativos. Toda fórmula analítica de gradiente debe validarse obligatoriamente mediante pruebas unitarias contra diferencias finitas centrales (`numerical_gradient`).
- **Fuente / experimento de verificación**:
  Prueba unitaria `tests/test_functions.py::TestRastrigin::test_gradient_vs_central_differences`.

---

## Hallazgo H-02
- **Autor que lo detectó**: Persona 1 (Funciones + Gradient Descent)
- **Fecha**: 2026-10-03
- **Categoría**: Matemáticas / Código
- **Prompt literal**:
  > *"Implementa el gradiente analítico general de la función de Rosenbrock para cualquier dimensión n en un solo vector numpy"*
- **Parte relevante de la respuesta de la IA**:
  ```python
  # Respuesta simplificada errónea de la IA:
  grad = np.zeros_like(x)
  for i in range(len(x) - 1):
      grad[i] = -400 * x[i] * (x[i+1] - x[i]**2) + 2 * (x[i] - 1)
  grad[-1] = 200 * (x[-1] - x[-2]**2)
  ```
- **Por qué sospechamos**:
  En dimensión $n \ge 3$, la variable intermedia $x_i$ ($2 \le i \le n-1$) aparece simultáneamente en dos términos consecutivos de la sumatoria:
  $100(x_i - x_{i-1}^2)^2$ (donde actúa como variable adelantada) y $100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2$ (donde actúa como variable actual). La IA omitió por completo el primer término para las componentes intermedias, dejando únicamente la fórmula correspondiente al caso 2D en cada posición.
- **Evidencia que demuestra el error**:
  Derivando la función general:
  $$f(x) = \sum_{j=1}^{n-1} \left[ 100(x_{j+1} - x_j^2)^2 + (1 - x_j)^2 \right]$$
  Para un índice intermedio $i \in \{2, \dots, n-1\}$, los únicos sumandos dependientes de $x_i$ son:
  $$S_{i-1} = 100(x_i - x_{i-1}^2)^2 \implies \frac{\partial S_{i-1}}{\partial x_i} = 200(x_i - x_{i-1}^2)$$
  $$S_i = 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \implies \frac{\partial S_i}{\partial x_i} = -400 x_i (x_{i+1} - x_i^2) + 2(x_i - 1)$$
  Por tanto, la derivada total es la suma de ambos:
  $$\frac{\partial f}{\partial x_i} = 200(x_i - x_{i-1}^2) - 400 x_i (x_{i+1} - x_i^2) + 2(x_i - 1)$$
  Al correr la prueba de diferencias centrales con $n=3$, la componente intermedia $x_2$ fallaba con un error absoluto de cientos de unidades cuando $x_2 \ne x_1^2$.
- **Corrección**:
  Separar explícitamente en tres casos vectorizados:
  1. $i = 0$ (primer elemento): $-400 x_1 (x_2 - x_1^2) + 2(x_1 - 1)$
  2. $1 \le i \le n-2$ (elementos intermedios): $200(x_i - x_{i-1}^2) - 400 x_i (x_{i+1} - x_i^2) + 2(x_i - 1)$
  3. $i = n-1$ (último elemento): $200(x_n - x_{n-1}^2)$
- **Lección aprendida**:
  En sumatorias acopladas o telescópicas ($x_{i+1}$ acoplado con $x_i$), los modelos de lenguaje tienden a derivar término a término de forma miope sin considerar que un mismo $x_i$ figura en dos sumandos distintos.
- **Fuente / experimento de verificación**:
  Prueba unitaria `tests/test_functions.py::TestRosenbrock::test_gradient_vs_central_differences` con parametrización $n=3$ y $n=5$.

---

## Hallazgo H-03
- **Autor que lo detectó**: Persona 1 (Funciones + Gradient Descent)
- **Fecha**: 2026-10-03
- **Categoría**: Conceptos / Código
- **Prompt literal**:
  > *"¿Cómo debo contar las evaluaciones para comparar Gradient Descent con Algoritmos Genéticos y PSO si tengo el gradiente analítico?"*
- **Parte relevante de la respuesta de la IA**:
  > *"Como el gradiente analítico se calcula directamente mediante una fórmula cerrada de numpy en un solo paso, cuenta como 1 sola evaluación o ni siquiera debe sumarse a las evaluaciones de la función objetivo f(x), pues no evalúa f(x)."*
- **Por qué sospechamos**:
  Esa afirmación destruye el principio de **comparación justa** del enunciado. Si se contara el gradiente como 0 o como 1 evaluación, Gradient Descent parecería converger en órdenes de magnitud menos esfuerzo computacional que las metaheurísticas de forma ficticia. En problemas reales de caja negra (*black-box*), el gradiente no está disponible analíticamente y debe aproximarse mediante diferencias finitas centrales, lo que requiere $2n$ evaluaciones de la función en dimensión $n$.
- **Evidencia que demuestra el error**:
  El enunciado oficial (pág. 2) y la Guía Operativa (Sección 2.4) estipulan explícitamente:
  > *"Propongan y justifiquen una equivalencia entre ambas. Por ejemplo, aproximar el gradiente por diferencias centrales cuesta 2n evaluaciones de f en dimensión n."*
  $$E_{eq} = N_f + 2n \cdot N_{grad}$$
- **Corrección**:
  Se implementó el wrapper `CountedFunction` (`src/metrics/evaluation_counter.py`) que cuenta $N_f$ y $N_{grad}$ de forma separada y estricta en tiempo de ejecución, calculando el costo equivalente $E_{eq} = N_f + 2n N_{grad}$ y bloqueando cualquier cálculo que supere el `budget`.
- **Lección aprendida**:
  Las IAs confunden con frecuencia la complejidad de tiempo de ejecución de una función cerrada en software con la complejidad teórica contable de evaluaciones exigida en bancos de pruebas (*benchmarks*) de optimización.
- **Fuente / experimento de verificación**:
  Prueba unitaria `tests/test_gradient_descent.py::TestGradientDescent::test_strict_budget_compliance`.
