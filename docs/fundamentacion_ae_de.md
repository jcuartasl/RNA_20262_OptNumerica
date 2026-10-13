# Fundamentación de los algoritmos evolutivos (AE) y evolución diferencial (DE)

## 1. Introducción

La optimización numérica busca encontrar un vector de variables que minimice o maximice una función objetivo. En este proyecto se comparan descenso por gradiente (GD), optimización por enjambre de partículas (PSO), algoritmos evolutivos (AE) y evolución diferencial (DE) sobre Rosenbrock y Rastrigin, en dos y tres dimensiones. Esta sección describe los mecanismos de AE y DE a partir del código disponible en el repositorio, y relaciona sus decisiones de implementación con los resultados del experimento.

Un algoritmo evolutivo mantiene una **población** de soluciones candidatas en lugar de modificar una única solución. Cada **individuo** es un vector de variables; su calidad se mide evaluando la función objetivo. La **selección** favorece candidatos con mejor valor objetivo, el **cruce** combina información de padres, la **mutación** introduce variaciones aleatorias y el **elitismo** conserva algunas de las mejores soluciones. Estos operadores no garantizan encontrar el óptimo global, pero permiten explorar regiones distintas del dominio sin requerir derivadas de la función (Eiben & Smith, 2015; Goldberg, 1989).

En este proyecto los dos métodos son de población y trabajan con vectores reales. No obstante, AE utiliza selección por torneo, cruce BLX-\(\alpha\) y mutación gaussiana; DE construye candidatos mediante diferencias entre miembros de la población y los combina con el vector objetivo. La distinción es relevante: aunque ambos utilizan aleatoriedad, generan sus candidatos de formas diferentes.

## 2. Algoritmo evolutivo (AE)

### 2.1. Representación e inicialización

Un individuo se representa como un vector real

\[
\mathbf{x}_i=(x_{i,1},x_{i,2},\ldots,x_{i,n})\in\mathbb{R}^n,
\]

donde \(n\) es la dimensión del problema y \(x_{i,j}\) es la coordenada \(j\) del individuo \(i\). La población reúne estos vectores en una matriz de tamaño \(N\times n\), donde \(N\) es el tamaño de población.

La implementación inicializa cada coordenada mediante una distribución uniforme entre los límites inferior y superior correspondientes. Para cada individuo:

\[
x_{i,j}\sim \mathcal{U}(l_j,u_j),
\]

donde \(l_j\) y \(u_j\) son los límites de la coordenada \(j\). La semilla se utiliza para crear un generador pseudoaleatorio (`numpy.random.default_rng`), por lo que repetir una ejecución con la misma semilla, función, presupuesto y configuración permite reproducir la secuencia aleatoria.

Cada individuo se evalúa una vez para obtener su aptitud \(f(\mathbf{x}_i)\). Como se trata de minimización, un valor menor es mejor. La función auxiliar de evaluación respeta el presupuesto disponible y puede devolver una población inicial incompleta si no hay evaluaciones suficientes. Si ninguna aptitud inicial es finita, la implementación genera un error en vez de continuar con una población sin soluciones utilizables.

### 2.2. Selección por torneo

Para elegir cada padre, `seleccion_torneo` toma al azar `tournament_size` índices distintos de la población y escoge al participante con la menor aptitud. Si el conjunto de participantes es \(T\), el ganador es

\[
i^*=\arg\min_{i\in T} f(\mathbf{x}_i).
\]

En la configuración del experimento, el torneo tiene tamaño 3. Un torneo de este tamaño impone presión selectiva: los individuos de mejor aptitud tienen más oportunidades de reproducirse, pero no se escoge siempre de manera directa al mejor de toda la población. El muestreo se realiza sin reemplazo dentro de cada torneo; torneos distintos pueden seleccionar al mismo individuo. Esto puede acelerar la mejora, aunque una presión selectiva demasiado alta puede reducir la diversidad.

### 2.3. Cruce BLX-\(\alpha\)

La implementación no usa el cruce aritmético de combinación ponderada que suele expresarse como \(\lambda\mathbf{x}_a+(1-\lambda)\mathbf{x}_b\). Utiliza **BLX-\(\alpha\)**, un cruce real que muestrea cada coordenada dentro de un intervalo ampliado alrededor de los dos padres.

Para los padres \(\mathbf{x}_a\) y \(\mathbf{x}_b\), en la coordenada \(j\) se define

\[
d_j=|x_{a,j}-x_{b,j}|,
\quad
L_j=\min(x_{a,j},x_{b,j})-\alpha d_j,
\quad
U_j=\max(x_{a,j},x_{b,j})+\alpha d_j.
\]

Si se activa el cruce, cada coordenada de cada hijo se genera de forma independiente:

\[
x_{h,j}\sim\mathcal{U}(L_j,U_j).
\]

Se generan dos hijos por pareja y ambos se muestrean de esos mismos intervalos, por lo que no son necesariamente complementarios. El cruce se activa cuando un número aleatorio uniforme es menor que `crossover_rate`; con probabilidad \(1-\text{crossover\_rate}\), los hijos son copias de los padres.

El parámetro \(\alpha\) determina cuánto se amplía el intervalo más allá de los valores parentales. Con \(\alpha=0\), las coordenadas quedan dentro del intervalo entre ambos padres. Con \(\alpha=0.5\), como en este experimento, el intervalo se amplía en la mitad de la distancia parental por cada extremo. Esta expansión puede producir soluciones fuera del rango de los padres y favorecer exploración, pero también puede generar candidatos alejados de regiones prometedoras. Por eso los hijos se ajustan después a los límites globales.

### 2.4. Mutación gaussiana

Tras el cruce, la implementación aplica mutación de manera independiente a cada coordenada. Se genera una máscara aleatoria y cada gen se muta si el número uniforme correspondiente es menor que `mutation_rate`. Para una coordenada mutada:

\[
x'_{i,j}=x_{i,j}+\epsilon_{i,j},
\qquad
\epsilon_{i,j}\sim\mathcal{N}(0,\sigma_j^2).
\]

Las coordenadas no seleccionadas por la máscara permanecen sin cambio. La desviación estándar se calcula de forma relativa al ancho de cada dimensión:

\[
\sigma_j=\text{mutation\_sigma\_rel}\,(u_j-l_j).
\]

En la configuración, `mutation_rate = 0.10`, es decir, cada coordenada tiene probabilidad del 10 % de recibir ruido, y `mutation_sigma_rel = 0.10`, de modo que la desviación estándar equivale al 10 % del ancho del dominio de esa coordenada. No significa que cada individuo se mute con probabilidad del 10 %: la decisión se toma por gen. Una mutación moderada puede recuperar diversidad o permitir salir de una región estancada; si el ruido es demasiado grande puede deshacer combinaciones útiles, y si es demasiado pequeño puede limitar la exploración.

### 2.5. Ajuste de límites, elitismo y formación de la generación

Después de mutar, los individuos se mantienen dentro del dominio mediante recorte por coordenada:

\[
x_{i,j}\leftarrow \min\{u_j,\max\{l_j,x_{i,j}\}\}.
\]

Este procedimiento, conocido como *clipping*, reemplaza cualquier valor por debajo o por encima del intervalo por el límite correspondiente. Es sencillo y evita evaluar candidatos fuera del dominio, aunque puede acumular individuos en las fronteras si los operadores producen valores extremos con frecuencia.

Antes de construir la siguiente generación, `elitismo` ordena las aptitudes y copia los mejores `n_elites` individuos junto con sus valores ya calculados. La implementación conserva dos élites. No los vuelve a evaluar: esto ahorra presupuesto y evita perder las mejores soluciones encontradas en la generación actual. El resto de la población se completa con hijos recién generados y evaluados. En forma esquemática:

\[
P_{t+1}=E_t\cup H_t,
\]

donde \(E_t\) es el conjunto de élites de la generación \(t\), y \(H_t\) son los hijos evaluados. En la última generación puede haber menos hijos de lo habitual si el presupuesto restante no alcanza para completar la generación.

La cantidad de padres se ajusta para que cada pareja pueda producir dos hijos; si se requiere un número impar de hijos, se genera un hijo adicional y luego se descarta el sobrante. La aptitud de los hijos se evalúa con el contador común. El mejor global se actualiza únicamente cuando aparece un valor objetivo menor que el mejor registrado hasta ese momento.

### 2.6. Parada y presupuesto

El ciclo termina cuando se agota el presupuesto de evaluaciones equivalentes o cuando se alcanza `max_generations`, si este parámetro tiene un entero configurado. En `config/part1.yaml`, `max_generations: null`, por lo que no se impone un límite de generaciones separado: la condición práctica es el presupuesto. La tolerancia de éxito global es \(10^{-4}\), y el resultado se marca exitoso si

\[
|f_{\mathrm{mejor}}-f^*|\leq 10^{-4},
\]

donde \(f^*\) es el mínimo global conocido de la función. La tolerancia se usa para etiquetar el resultado; el AE no termina anticipadamente solo por alcanzar ese valor, sino que continúa mientras tenga presupuesto.

El presupuesto se comprueba antes de cada evaluación de función. La población inicial puede reducirse si el presupuesto disponible es menor que el tamaño configurado. En las generaciones posteriores, las élites pasan sin reevaluación y solo se cobra la evaluación de los hijos. La cantidad de hijos se limita al presupuesto restante. El historial registra el mejor valor, la solución, el número de evaluaciones de función y las evaluaciones equivalentes; la población completa solo se registra si `record_population` está activado, y en la configuración está desactivado.

### 2.7. Parámetros de AE

Los valores configurados en `config/part1.yaml` y el comportamiento del código se resumen en la tabla.

| Parámetro | Valor configurado | Significado y efecto esperado |
|---|---:|---|
| `population_size` | 50 | Número objetivo de individuos. Una población mayor suele representar más regiones, pero consume más evaluaciones en la inicialización. |
| `tournament_size` | 3 | Número de participantes por torneo. Favorece aptitudes bajas sin seleccionar siempre al mejor global. |
| `crossover_rate` | 0.90 | Probabilidad de activar BLX-\(\alpha\) para una pareja. Si no se activa, se copian los padres. |
| `alpha` | 0.5 | Amplía el intervalo de muestreo de BLX-\(\alpha\). Valores mayores permiten hijos más alejados de los padres. |
| `mutation_rate` | 0.10 | Probabilidad independiente de mutación por coordenada, no por individuo. |
| `mutation_sigma_rel` | 0.10 | Multiplica el ancho del dominio para obtener la desviación estándar del ruido gaussiano en cada coordenada. |
| `n_elites` | 2 | Número de mejores individuos que se copian sin reevaluación a la siguiente generación. |
| `max_generations` | `null` | No impone un máximo de generaciones; la ejecución queda limitada por el presupuesto. |
| `record_population` | `false` | No guarda la matriz completa de población en cada entrada del historial. |
| `tolerance` | \(10^{-4}\) | Criterio de éxito comparando la mejor aptitud con el mínimo global conocido. Está configurado a nivel del experimento; el código de AE lo valida/usa mediante su configuración. |
| `population_size` mínimo / validaciones | Según `validar_configuracion` | Las validaciones adicionales dependen del código; no son hiperparámetros nuevos del experimento. |

La tabla separa los valores explícitos de la configuración de los criterios experimentales. La implementación también valida la dimensión, la semilla, el presupuesto y los límites antes de ejecutar. Los detalles de validación de cada campo de configuración están encapsulados en `validar_configuracion`.

## 3. Evolución diferencial (DE)

### 3.1. Variante implementada y representación

El archivo `src/optimizers/differential_evolution.py` declara e implementa la variante **DE/rand/1/bin**, con minimización y actualización síncrona. La representación de cada individuo es un vector real \(\mathbf{x}_i\in\mathbb{R}^n\), y la población inicial tiene tamaño \(NP=50\). Cada coordenada se muestrea uniformemente dentro de sus límites y se evalúa hasta donde permita el presupuesto.

La variante se identifica por tres decisiones: `rand` indica que la base de la mutación se escoge aleatoriamente entre individuos de la población; `1` indica que se añade un vector de diferencia escalado; `bin` indica que el cruce es binomial. Esta notación corresponde a la formulación clásica de evolución diferencial (Storn & Price, 1997).

### 3.2. Mutación diferencial

Para cada vector objetivo \(\mathbf{x}_i\), el código selecciona tres índices \(r_1,r_2,r_3\) distintos entre sí y diferentes de \(i\). La elección se hace sin reemplazo entre los índices disponibles. El vector mutante se calcula como

\[
\mathbf{v}_i=\mathbf{x}_{r_1}+F(\mathbf{x}_{r_2}-\mathbf{x}_{r_3}),
\]

donde \(F\) es el factor de mutación, configurado en \(0.80\). La diferencia entre dos individuos proporciona una dirección y una escala relacionadas con la dispersión actual de la población; \(F\) controla cuánto pesa esa diferencia. Un valor mayor puede ampliar los saltos y explorar más, pero también producir propuestas difíciles de aceptar. Un valor pequeño genera variaciones más contenidas y puede limitar la exploración.

Antes del cruce, el vector mutante se recorta a los límites del dominio. Por tanto, la implementación aplica el ajuste de límites al mutante, no después de construir el vector de prueba. Como el vector de prueba combina coordenadas del mutante ya recortado y del objetivo que ya pertenece al dominio, también queda dentro de los límites.

### 3.3. Cruce binomial

El vector de prueba \(\mathbf{u}_i\) combina el vector mutante \(\mathbf{v}_i\) con el vector objetivo \(\mathbf{x}_i\). Para cada coordenada se genera un número aleatorio; se toma la coordenada del mutante si ese número es menor que \(CR\), o si corresponde a un índice especial \(j_{\mathrm{rand}}\). En símbolos:

\[
u_{i,j}=
\begin{cases}
v_{i,j}, & \text{si } r_j<CR\text{ o }j=j_{\mathrm{rand}},\\
x_{i,j}, & \text{en otro caso},
\end{cases}
\]

donde \(r_j\sim\mathcal{U}(0,1)\), \(CR\) es la tasa de cruce y \(j_{\mathrm{rand}}\) es una coordenada elegida aleatoriamente. Esta coordenada especial garantiza que al menos un gen proceda del mutante incluso cuando \(CR=0\). En el experimento, \(CR=0.90\), así que suele heredarse una parte importante del mutante, aunque la coordenada forzada es la garantía formal de que el candidato no sea una copia exacta del objetivo.

### 3.4. Selección voraz y actualización síncrona

Después de evaluar el vector de prueba, se compara con el objetivo:

\[
\mathbf{x}_i^{(t+1)}=
\begin{cases}
\mathbf{u}_i, & f(\mathbf{u}_i)\leq f(\mathbf{x}_i^{(t)}),\\
\mathbf{x}_i^{(t)}, & \text{en otro caso}.
\end{cases}
\]

La igualdad se acepta, tal como indica el uso de `<=` en el código. La selección es voraz porque cada candidato compite únicamente con el individuo del que se generó. La implementación es síncrona: todos los vectores de prueba de una generación se construyen usando la población vigente al inicio de esa generación; las sustituciones se guardan en `new_pop` y se aplican al final del ciclo. Esto evita que un candidato recién aceptado altere las propuestas de los individuos que todavía no se han procesado en esa misma generación.

El algoritmo no se detiene anticipadamente por alcanzar la tolerancia de éxito. Sigue consumiendo presupuesto mientras pueda evaluar candidatos. Si el presupuesto no alcanza para evaluar la población inicial completa, evalúa los individuos que caben y no inicia el ciclo evolutivo. En una última generación puede procesar solo parte de los individuos. `iterations` cuenta las generaciones iniciadas, incluida una última generación parcial.

### 3.5. Parámetros de DE

| Parámetro | Valor configurado | Significado y efecto esperado |
|---|---:|---|
| `population_size` | 50 | Tamaño \(NP\) de la población. La mutación `rand/1` exige al menos cuatro individuos; el código valida \(NP\geq4\). |
| `mutation_factor` | 0.80 | Factor \(F\) que escala la diferencia entre dos vectores en la mutación. Debe ser positivo. |
| `crossover_rate` | 0.90 | Probabilidad \(CR\) de tomar coordenadas del mutante; el código exige \(0\leq CR\leq1\). Siempre se fuerza una coordenada mutante. |
| `tolerance` | \(10^{-4}\) | Umbral para marcar éxito respecto al mínimo global conocido. Procede de la configuración experimental o, si no se pasa, del valor predeterminado del optimizador. |

En el código, los valores predeterminados de DE son \(NP=50\), \(F=0.8\), \(CR=0.9\) y tolerancia \(10^{-4}\). Los tres primeros también aparecen explícitamente en `config/part1.yaml`; la tolerancia aparece en la sección general `experiment`. El optimizador acepta `config` opcional y usa esos valores predeterminados cuando no se suministra el campo correspondiente.

## 4. Presupuesto computacional y comparación justa

Contar generaciones no permite comparar directamente algoritmos distintos: una generación de DE evalúa potencialmente un candidato por individuo, mientras que GD calcula gradientes y evalúa puntos nuevos, con un costo distinto. El proyecto utiliza `CountedFunction`, definido en `src/metrics/evaluation_counter.py`, para registrar las evaluaciones de función \(N_f\) y las evaluaciones de gradiente \(N_{\nabla f}\). El contador define las evaluaciones equivalentes como

\[
E_{\mathrm{eq}}=N_f+2nN_{\nabla f},
\]

donde \(n\) es la dimensión. Cada llamada a la función objetivo cuesta una unidad y cada evaluación del gradiente cuesta \(2n\) unidades equivalentes. El método `can_evaluate_gradient` comprueba precisamente si ese costo adicional cabe en el presupuesto. En GD, una iteración completa requiere un gradiente y una evaluación de la función en el nuevo punto, por lo que el costo presupuestado es \(2n+1\). AE, PSO y DE no usan gradientes en sus optimizadores y, por tanto, tienen \(N_{\nabla f}=0\) y \(E_{\mathrm{eq}}=N_f\).

El presupuesto configurado es de 10 000 evaluaciones equivalentes por ejecución. Los optimizadores consultan el contador antes de evaluar y ajustan la última población o generación cuando es necesario, para no superar ese límite. Por eso las generaciones realizadas pueden variar entre métodos y dimensiones; la comparación relevante se hace bajo el mismo presupuesto, no igualando artificialmente el número de iteraciones.

El experimento define 30 semillas, desde 0 hasta 29, para cada combinación de algoritmo, función y dimensión. Con cuatro algoritmos, dos funciones y dos dimensiones se obtienen \(4\times2\times2=16\) configuraciones y \(16\times30=480\) ejecuciones previstas. Usar el mismo conjunto de semillas facilita una comparación reproducible y reduce diferencias atribuibles únicamente a haber elegido conjuntos de inicializaciones distintos. No elimina toda variabilidad, ni implica que los algoritmos compartan exactamente las mismas posiciones iniciales: cada método consume los números aleatorios de acuerdo con su propio mecanismo. Las 30 repeticiones permiten observar con qué frecuencia se alcanza el criterio de éxito, aunque para inferencias más amplias harían falta análisis estadísticos y más problemas.

## 5. Funciones de prueba

Rosenbrock y Rastrigin permiten observar comportamientos diferentes en problemas continuos de dimensión baja.

La función de Rosenbrock usada habitualmente en este proyecto se expresa como

\[
f(\mathbf{x})=\sum_{i=1}^{n-1}\left[100(x_{i+1}-x_i^2)^2+(1-x_i)^2\right].
\]

Su mínimo global es \(f(\mathbf{x}^*)=0\) en \(\mathbf{x}^*=(1,\ldots,1)\). El valle que conduce al mínimo es estrecho y curvado. Un método puede avanzar hacia la región correcta y, aun así, tener dificultad para seguir el valle hasta el punto óptimo. En métodos poblacionales, la diversidad puede ayudar a explorar distintas zonas, pero no garantiza una refinación precisa.

La función de Rastrigin se define como

\[
f(\mathbf{x})=10n+\sum_{i=1}^{n}\left[x_i^2-10\cos(2\pi x_i)\right],
\]

con mínimo global \(f(\mathbf{0})=0\). Su paisaje tiene numerosos mínimos locales debidos al término coseno. Por ello, un algoritmo puede encontrar regiones de baja aptitud sin llegar al mínimo global. La exploración poblacional y la capacidad de generar variaciones son especialmente relevantes en este caso.

Las fórmulas anteriores corresponden a las definiciones estándar. La copia del repositorio incluida en el ZIP contiene `src/functions/rosenbrock.py` y `src/functions/rastrigin.py`; conviene mantener esas implementaciones como referencia definitiva si se modifica la definición o el escalamiento de las funciones.

## 6. Resultados experimentales y análisis

La tabla siguiente resume las tasas de éxito reportadas para las 30 semillas de cada configuración.

| Función | Dimensión | GD | PSO | DE | AE |
|---|---:|---:|---:|---:|---:|
| Rosenbrock | 2 | 0 % | 100 % | 100 % | 16,67 % |
| Rosenbrock | 3 | 0 % | 6,67 % | 100 % | 6,67 % |
| Rastrigin | 2 | 0 % | 100 % | 100 % | 100 % |
| Rastrigin | 3 | 0 % | 96,67 % | 83,33 % | 96,67 % |

Estas cifras deben leerse como resultados de las configuraciones, semillas, límites y presupuesto utilizados, no como una clasificación universal de los algoritmos. En las ejecuciones reportadas, DE alcanzó el 100 % de éxito en Rosenbrock para ambas dimensiones y en Rastrigin de dos dimensiones; en Rastrigin de tres dimensiones obtuvo 83,33 %. AE alcanzó el 100 % en Rastrigin de dos dimensiones y 96,67 % en tres dimensiones, pero sus tasas fueron menores en Rosenbrock: 16,67 % en dos variables y 6,67 % en tres.

Una posible interpretación es que la diferencia vectorial de DE produce desplazamientos ligados a la distribución de la población, mientras que AE depende de la combinación BLX-\(\alpha\), la mutación gaussiana y el torneo. En Rosenbrock, el valle estrecho y curvado puede hacer que el comportamiento de los operadores y la escala de los cambios sean relevantes para llegar con precisión al mínimo. En Rastrigin, la estructura multimodal plantea el reto de escapar de mínimos locales; los resultados muestran un buen desempeño de AE en las dimensiones ensayadas y un descenso de la tasa de éxito de DE al pasar a tres variables. Sin registros adicionales y un análisis específico no se puede atribuir esa diferencia a un único operador ni afirmar una causa definitiva.

GD presenta 0 % de éxito en las cuatro configuraciones reportadas. Esto indica que, con la inicialización, tasa de aprendizaje, tolerancia de gradiente y presupuesto utilizados, no alcanzó el umbral de éxito en estas ejecuciones. No demuestra que GD sea inadecuado para toda función ni que los métodos poblacionales siempre lo superen. Además, el ZIP proporcionado no contiene `results/summary/parte1_comparacion.csv` ni las gráficas consolidadas solicitadas; por eso no se añaden medias, dispersiones, tiempos ni pruebas estadísticas que no estén disponibles en los archivos recibidos.

## 7. Comparación entre AE y DE

| Aspecto | AE implementado | DE implementado |
|---|---|---|
| Generación de candidatos | Selección por torneo, cruce BLX-\(\alpha\) y mutación gaussiana. | Mutación a partir de un individuo base y una diferencia entre otros dos; después, cruce binomial. |
| Uso de la población | Los padres se eligen por aptitud y sus coordenadas se combinan mediante intervalos BLX-\(\alpha\). | Las diferencias entre vectores aportan dirección y escala al vector mutante. |
| Mutación | Ruido gaussiano independiente por coordenada, con probabilidad `mutation_rate`. | Diferencia vectorial escalada por \(F\); no es ruido gaussiano independiente. |
| Selección | Torneo para elegir padres; las élites sobreviven y la siguiente población se forma con élites e hijos. | Selección voraz uno a uno: el vector de prueba reemplaza al objetivo si su aptitud no es peor. |
| Conservación de soluciones | Elitismo explícito de dos individuos. | La selección voraz conserva cada objetivo si el candidato no lo mejora; no hay un bloque separado de élites. |
| Diversidad | Depende del torneo, la amplitud de BLX-\(\alpha\), la tasa y escala de mutación. | Depende de la dispersión poblacional, \(F\), \(CR\) y la selección voraz. |
| Parámetros principales | Tamaño poblacional, torneo, tasa de cruce, \(\alpha\), tasa y escala de mutación, élites. | Tamaño poblacional, factor \(F\), tasa \(CR\). |
| Límites | Recorte de hijos tras cruce y mutación. | Recorte del vector mutante antes del cruce. |
| Actualización | Nueva población de élites más hijos; los hijos se evalúan antes de incorporarse. | Actualización síncrona: todos los candidatos de una generación usan la población de la generación anterior. |
| Posible ventaja | Operadores distintos permiten controlar por separado selección, recombinación y variación. | Pocos parámetros principales y escala de búsqueda adaptada a diferencias presentes en la población. |
| Posible limitación | Puede perder diversidad o generar saltos poco adecuados si los parámetros no se ajustan a la función. | Si la población pierde diversidad, los vectores diferencia pueden volverse pequeños; el rendimiento es sensible a \(F\), \(CR\) y \(NP\). |

En la práctica, AE ofrece un control explícito de la presión selectiva, la recombinación, la mutación y la conservación de élites. DE tiene una mecánica de generación de candidatos más compacta: utiliza diferencias entre soluciones como fuente de variación y decide si cada candidato sustituye a su objetivo. Ambos son estocásticos y sensibles a la configuración, pero no conviene describir el cruce de AE como una combinación aritmética ni la mutación de DE como ruido gaussiano, porque esas no son las operaciones implementadas.

## 8. Conclusiones

AE y DE resuelven el problema mediante poblaciones, pero se diferencian en la forma de producir y seleccionar candidatos. El AE del repositorio usa torneo, cruce BLX-\(\alpha\), mutación gaussiana por coordenada y elitismo de dos individuos. DE implementa `DE/rand/1/bin`, con mutación diferencial, cruce binomial y selección voraz síncrona.

En las ejecuciones reportadas, DE obtuvo tasas altas en Rosenbrock y Rastrigin de dos dimensiones, mientras que AE tuvo un desempeño alto en Rastrigin y más limitado en Rosenbrock. Estos resultados son evidencia del comportamiento bajo el protocolo probado, no una demostración de superioridad general. La comparación por 10 000 evaluaciones equivalentes y el uso de 30 semillas por configuración hacen más consistente el experimento, aunque no eliminan la sensibilidad a los parámetros ni reemplazan el análisis estadístico.

Finalmente, las conclusiones están limitadas a dos funciones, dos dimensiones y una configuración concreta por algoritmo. Para generalizar sería necesario ampliar el conjunto de problemas, explorar parámetros y reportar métricas adicionales, manteniendo el mismo criterio de presupuesto.

## Referencias

Eiben, A. E., & Smith, J. E. (2015). *Introduction to evolutionary computing* (2nd ed.). Springer. https://doi.org/10.1007/978-3-662-44874-8

Goldberg, D. E. (1989). *Genetic algorithms in search, optimization, and machine learning*. Addison-Wesley.

Storn, R., & Price, K. (1997). Differential evolution—A simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization, 11*(4), 341–359. https://doi.org/10.1023/A:1008202821328
