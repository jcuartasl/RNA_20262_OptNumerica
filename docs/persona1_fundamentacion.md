# Fundamentación Matemática y Aporte Técnico: Persona 1
## Funciones de Prueba, Gradientes Analíticos y Descenso por Gradiente

**Autor:** Persona 1 (Funciones + Gradient Descent)  
**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Semestre:** 2026-02 — Universidad Nacional de Colombia  

---

## 1. Fundamentación Matemática de las Funciones de Prueba

### 1.1 Función de Rosenbrock (Valle de Rosenbrock / Banana Function)

La función de Rosenbrock es un problema clásico de prueba no convexo propuesto por Rosenbrock (1960). En dimensión $n \ge 2$, se define como:

$$f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100 \left( x_{i+1} - x_i^2 \right)^2 + (1 - x_i)^2 \right]$$

- **Dominio experimental:** $\mathbf{x} \in [-5.0, 10.0]^n$ (Surjanovic & Bingham, 2013).
- **Mínimo global:** $\mathbf{x}^* = (1, 1, \dots, 1)^T$, con valor óptimo $f(\mathbf{x}^*) = 0.0$ (Rosenbrock, 1960; Surjanovic & Bingham, 2013).

#### Derivación analítica del gradiente $\nabla f(\mathbf{x})$:
Para un vector $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$, los términos de la sumatoria que dependen de la coordenada $x_i$ cambian según el índice:

1. **Primera componente ($i = 1$):**  
   Solo interviene en el primer sumando ($j=1$):
   $$\frac{\partial f}{\partial x_1} = 100 \cdot 2(x_2 - x_1^2)(-2x_1) + 2(1 - x_1)(-1) = -400 x_1 (x_2 - x_1^2) + 2(x_1 - 1)$$

2. **Componentes intermedias ($2 \le i \le n-1$):**  
   La coordenada $x_i$ aparece en dos sumandos consecutivos: en $j = i-1$ como $100(x_i - x_{i-1}^2)^2$ y en $j = i$ como $100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2$. Sumando ambas derivadas parciales:
   $$\frac{\partial f}{\partial x_i} = 200(x_i - x_{i-1}^2) - 400 x_i (x_{i+1} - x_i^2) + 2(x_i - 1)$$

3. **Última componente ($i = n$):**  
   Solo interviene en el último sumando ($j = n-1$), $100(x_n - x_{n-1}^2)^2$:
   $$\frac{\partial f}{\partial x_n} = 200(x_n - x_{n-1}^2)$$

En el mínimo global $\mathbf{x}^* = (1, \dots, 1)^T$, se cumple $x_{i+1} - x_i^2 = 0$ y $x_i - 1 = 0$, verificándose que $\nabla f(\mathbf{x}^*) = \mathbf{0}$.

---

### 1.2 Función de Rastrigin

Propuesta originalmente por Rastrigin (1974) y extendida como banco de pruebas multidimensional por Mühlenbein et al. (1991), es una función altamente multimodal obtenida al añadir una modulación cosenoidal a una parábola esférica:

$$f(\mathbf{x}) = 10n + \sum_{i=1}^n \left[ x_i^2 - 10 \cos(2\pi x_i) \right]$$

- **Dominio experimental:** $\mathbf{x} \in [-5.12, 5.12]^n$ (Mühlenbein et al., 1991; Surjanovic & Bingham, 2013).
- **Mínimo global único:** $\mathbf{x}^* = (0, 0, \dots, 0)^T$, con valor óptimo $f(\mathbf{x}^*) = 0.0$.

#### Derivación analítica del gradiente $\nabla f(\mathbf{x})$:
Al ser una función separable por coordenadas, la derivada parcial respecto a cada componente $x_i$ es:

$$\frac{\partial f}{\partial x_i} = \frac{d}{dx_i} \left[ x_i^2 - 10 \cos(2\pi x_i) \right] = 2x_i - 10 \left( -\sin(2\pi x_i) \cdot 2\pi \right) = 2x_i + 20\pi \sin(2\pi x_i)$$

En el óptimo $\mathbf{x}^* = \mathbf{0}$, cada componente evalúa $2(0) + 20\pi \sin(0) = 0$, por lo que $\nabla f(\mathbf{x}^*) = \mathbf{0}$.

---

## 2. Análisis Geométrico y Comportamiento del Descenso por Gradiente

### 2.1 ¿Por qué Rosenbrock es difícil para el Descenso por Gradiente?
1. **Valle parabólico estrecho:** La superficie desciende rápidamente hacia el valle curvo $x_{i+1} \approx x_i^2$, pero dentro del valle la pendiente longitudinal hacia $\mathbf{x}^* = (1, \dots, 1)^T$ es muy suave (Surjanovic & Bingham, 2013).
2. **Mal condicionamiento de la matriz Hessiana ($\kappa(\nabla^2 f) \gg 1$):**  
   En el óptimo $\mathbf{x}^* = (1, 1)^T$ en 2D, la matriz Hessiana es:
   $$\nabla^2 f(1, 1) = \begin{bmatrix} 802 & -400 \\ -400 & 200 \end{bmatrix}$$
   cuyos autovalores son $\lambda_{\max} \approx 1001.6$ y $\lambda_{\min} \approx 0.399$, lo que arroja un número de condición $\kappa = \lambda_{\max}/\lambda_{\min} \approx 2508$ (Nocedal & Wright, 2006). Lejos del óptimo, hacia los extremos del dominio $[-5, 10]^n$, la magnitud del gradiente supera $3.6 \times 10^5$.
3. **Inestabilidad con paso fijo vs. lentitud en el valle:**  
   Un tamaño de paso constante $\alpha = 10^{-3}$ provoca saltos de cientos de unidades cuando $\mathbf{x}_0$ cae lejos del valle, haciendo que el *clipping* proyecte el iterado contra las esquinas del dominio y el algoritmo quede atrapado rebotando entre fronteras. Si se reduce el paso a $\alpha \le 10^{-4}$ para garantizar descenso estable hacia el valle, el avance a lo largo del fondo del valle (gobernado por $\lambda_{\min}$) se vuelve tan lento que 10,000 evaluaciones equivalentes no bastan para alcanzar la tolerancia de $10^{-4}$.

### 2.2 ¿Por qué Rastrigin atrapa al Descenso por Gradiente?
1. **Cuadrícula densa de mínimos locales ($11^n$ mínimos locales en el dominio):**  
   Los puntos estacionarios satisfacen $\frac{\partial f}{\partial x_i} = 2x_i + 20\pi \sin(2\pi x_i) = 0 \implies \sin(2\pi x_i) = -\frac{x_i}{10\pi}$, con segunda derivada $\frac{\partial^2 f}{\partial x_i^2} = 2 + 40\pi^2 \cos(2\pi x_i)$. Cerca de cada entero $k \in \{-5, -4, \dots, 4, 5\}$ existe una raíz donde $\cos(2\pi x_i) \approx 1 > 0$ (curvatura positiva). Al haber $11$ mínimos locales por cada eje coordenado dentro de $[-5.12, 5.12]$, el producto cartesiano genera $11^n$ mínimos locales en dimensión $n$ ($121$ en 2D y $1331$ en 3D).
2. **Constante de Lipschitz y límite de estabilidad:**  
   La curvatura máxima es $L = \max \|\nabla^2 f\| = 2 + 40\pi^2 \approx 396.78$. Para que el descenso por gradiente con paso fijo converja al mínimo de su cuenca local sin oscilar, requiere $\alpha < \frac{2}{L} \approx 0.00504$ (Nocedal & Wright, 2006).
3. **Atrapamiento local:**  
   Con $\alpha \le 10^{-3}$, el algoritmo converge de forma rápida y determinista al mínimo local de la cuenca donde cayó $\mathbf{x}_0$ (cuyas coordenadas son cercanas al vector de enteros más próximo a $\mathbf{x}_0$). Solo si $\mathbf{x}_0 \in (-0.5, 0.5)^n$ —lo cual ocurre con probabilidad $(1/10.24)^2 \approx 0.95\%$ en 2D y $(1/10.24)^3 \approx 0.09\%$ en 3D— la cuenca local coincide con el mínimo global $\mathbf{x}^* = \mathbf{0}$.

---

## 3. Algoritmo de Descenso por Gradiente y Comparación Justa

### 3.1 Ecuación de Actualización y Política de Límites
Dado $\mathbf{x}_0 \sim U(\mathbf{l}, \mathbf{u})$ generado de manera reproducible con la semilla (`seed`) de la corrida, cada iteración $t$ ejecuta:

$$\mathbf{x}_{t+1} = \operatorname{clip}\left( \mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t), \, \mathbf{l}, \, \mathbf{u} \right), \quad \text{donde } \operatorname{clip}(x_i, l_i, u_i) = \min(\max(x_i, l_i), u_i)$$

### 3.2 Modelo de Costo Equivalente y Control de Presupuesto
De acuerdo con el contrato de comparación justa:

$$E_{eq} = N_f + 2n \cdot N_{grad}$$

- **Justificación del factor $2n$:** Aproximar las $n$ componentes de $\nabla f(\mathbf{x})$ mediante diferencias finitas centrales:
  $$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h \mathbf{e}_i) - f(\mathbf{x} - h \mathbf{e}_i)}{2h}$$
  requiere $2n$ evaluaciones de $f$.
- **Costo por iteración ($2n + 1$):** El algoritmo realiza 1 evaluación inicial $f(\mathbf{x}_0)$ y, en cada paso $t$, calcula $\nabla f(\mathbf{x}_t)$ (costo $2n$) y evalúa $f(\mathbf{x}_{t+1})$ (costo $1$) para actualizar el mejor valor histórico `best_f` y registrar la curva de convergencia en `history`.
- **Prevención de desperdicio al agotar presupuesto:** Antes de iniciar una iteración, se comprueba que el presupuesto restante cubra la iteración completa (`E_eq + 2n + 1 <= budget`). Así nunca se calcula ni se cobra un gradiente cuyo paso posterior no pueda evaluarse. Sin parada temprana por norma de gradiente, el número de iteraciones ejecutadas para un presupuesto $B$ es exactamente $k = \lfloor (B - 1) / (2n + 1) \rfloor$.

### 3.3 Resultados de Pruebas Piloto sobre `learning_rate` (30 semillas, $B = 10{,}000$)

| Función y Dimensión | $\alpha = 10^{-2}$ (mediana `best_f`) | $\alpha = 10^{-3}$ (mediana `best_f`) | $\alpha = 10^{-4}$ (mediana `best_f`) | $\alpha = 10^{-5}$ (mediana `best_f`) | Observación técnica |
|---|---|---|---|---|---|
| **Rosenbrock 2D** | $1.81 \times 10^4$ (50% en frontera) | $1.81 \times 10^4$ (50% en frontera) | **$2.37$** (0% en frontera) | **$1.61$** (0% en frontera) | Con $\alpha \ge 10^{-3}$ rebota entre esquinas; con $\alpha \le 10^{-4}$ entra al valle sin divergir. |
| **Rosenbrock 3D** | $1.13 \times 10^5$ (30% en frontera) | $9.97 \times 10^4$ (40% en frontera) | **$1.67$** (0% en frontera) | $3.54$ (0% en frontera) | $\alpha = 10^{-4}$ logra el mejor compromiso entre estabilidad y velocidad de descenso. |
| **Rastrigin 2D** | $12.93$ ($\alpha > 2/L$, oscila) | **$16.42$** (converge a mín. local) | $16.42$ (converge a mín. local) | $16.42$ (converge a mín. local) | Con $\alpha \le 10^{-3}$ converge limpiamente al pozo local de $\mathbf{x}_0$. |
| **Rastrigin 3D** | $21.18$ ($\alpha > 2/L$, oscila) | **$24.38$** (converge a mín. local) | $24.38$ (converge a mín. local) | $24.38$ (converge a mín. local) | Invariante para $\alpha \in [10^{-5}, 10^{-3}]$ porque cae en el mismo mínimo local. |

---

## 4. Guía para el Video de Contribución Individual (Persona 1)

El video debe durar entre **2 y 3 minutos**, en primera persona y mostrando evidencia real:

1. **Presentación (15 s):** Nombre y rol de **Persona 1: Funciones de Prueba, Gradientes y Descenso por Gradiente**.
2. **Código de Funciones y Gradientes (45 s):**
   - Mostrar [`src/functions/rosenbrock.py`](file:///c:/Users/USER/Documents/Universidad/RNA/RNA_20262_OptNumerica/src/functions/rosenbrock.py) y explicar por qué el gradiente para $n \ge 3$ distingue tres casos ($i=1$, intermedios $2 \le i \le n-1$ con término acoplado $200(x_i - x_{i-1}^2)$, y $i=n$).
   - Mostrar [`src/functions/rastrigin.py`](file:///c:/Users/USER/Documents/Universidad/RNA/RNA_20262_OptNumerica/src/functions/rastrigin.py) y señalar el signo $+20\pi \sin(2\pi x_i)$.
3. **Optimizador y Conteo Exacto (45 s):**
   - Mostrar el bucle de [`src/optimizers/gradient_descent.py`](file:///c:/Users/USER/Documents/Universidad/RNA/RNA_20262_OptNumerica/src/optimizers/gradient_descent.py): la verificación `counter.equivalent_evaluations + iteration_cost > budget` (costo $2n+1$ por paso) y el *clipping* a los límites.
4. **Pruebas y Hallazgos de Alucinación (45 s):**
   - Ejecutar `pytest -v` en la terminal mostrando las **77 pruebas pasando**.
   - Explicar uno de los hallazgos reales documentados en [`docs/hallucination_log.md`](file:///c:/Users/USER/Documents/Universidad/RNA/RNA_20262_OptNumerica/docs/hallucination_log.md): por ejemplo, el **H-04** (cómo el código inicial cobraba un gradiente final que no alcanzaba a dar el paso y cómo `test_budget_used_exactly` lo detectó) o el **H-02** (el rebote entre esquinas en Rosenbrock con $\alpha = 10^{-3}$).

---

## 5. Referencias Bibliográficas (Formato APA 7.ª Edición)

- Mühlenbein, H., Schomisch, D., & Born, J. (1991). The parallel genetic algorithm as a global optimizer. *Parallel Computing*, 17(6–7), 619–632. https://doi.org/10.1016/S0167-8191(05)80052-3
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5
- Rastrigin, L. A. (1974). *Systems of extremal control*. Nauka.
- Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a function. *The Computer Journal*, 3(3), 175–184. https://doi.org/10.1093/comjnl/3.3.175
- Surjanovic, S., & Bingham, D. (2013). *Virtual library of simulation experiments: Test functions and datasets*. Simon Fraser University. https://www.sfu.ca/~ssurjano/optimization.html
