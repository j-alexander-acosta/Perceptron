# Contexto 8: Integración de la Unidad III "Descenso por Gradiente" en la Suite Educativa
**Fecha de registro:** 2026-11-02

## 1. Instrucciones
* Agregar la nueva sección correspondiente a la presentación **"Descenso por Gradiente"** (*Ciencia de Datos — UCN, basado en material de CMU 11-785 de B. Raj et al. y Stanford CS221 de P. Liang et al.*, 47 diapositivas) en la aplicación educativa (tanto en la interfaz web interactiva `web/index.html` como en un módulo ejecutable en Python `descenso_gradiente.py`, el lanzador central `perceptron_xor.py` y la documentación `README.md` / `INFORME.md`).
* Mantener la estructura modular por **Unidades Académicas**, incorporando la **Unidad III: Descenso por Gradiente y Optimización** junto a las Unidades I (*Aproximación Universal*) y II (*Aprendizaje de Redes Neuronales*).
* Cubrir íntegramente los 5 ejes temáticos y demostraciones matemáticas de la cátedra:
  1. **El problema detrás de todo entrenamiento: minimizar una función (Diaps. 1–8):** Ubicación en modelos de reflejos, minimización del riesgo empírico $\min_{\mathbf{w}} \text{TrainLoss}(\mathbf{w})$, superficie de pérdida y curvas de nivel, y mapa del terreno 1D (mínimo global, mínimo local, máximo global y punto de inflexión con derivada nula).
  2. **Derivadas y gradiente: la dirección de máximo crecimiento (Diaps. 9–16):** Aproximación lineal local $\Delta y \approx f'(x)\Delta x$, derivadas parciales y vector gradiente $\nabla_{\mathbf{x}} f$ ($\Delta y \approx \nabla_{\mathbf{x}} f \cdot \Delta\mathbf{x}$), propiedad del producto punto ($\mathbf{u}\cdot\mathbf{v} = |\mathbf{u}||\mathbf{v}|\cos\theta$), **Propiedad 1** (dirección de máximo crecimiento $+\nabla f$ y máximo descenso $-\nabla f$), **Propiedad 2** (perpendicularidad del gradiente a las curvas de nivel) y las 3 afirmaciones de la Diapositiva 16.
  3. **¿Es de verdad un mínimo? Segunda derivada y Hessiano (Diaps. 17–25):** Puntos críticos ($\nabla f = \mathbf{0}$), criterio de segunda derivada en 1D ($f(x)=x^2-4x+5 \implies x^*=2, f''(2)=2>0$), matriz Hessiana $\nabla_{\mathbf{x}}^2 f$ en $\mathbb{R}^d$ (definida positiva $\Rightarrow$ mínimo local, definida negativa $\Rightarrow$ máximo local, valores propios mixtos $\Rightarrow$ punto de silla) y resolución completa del ejemplo en $\mathbb{R}^3$ de las Diapositivas 24–25:
     $$f(x_1, x_2, x_3) = x_1^2 + x_1(1 - x_2) + x_2^2 - x_2 x_3 + x_3^2 + x_3 \implies \mathbf{x}^* = (-1, -1, -1), \quad \lambda = \{3.414, 0.586, 2\}$$
  4. **Soluciones iterativas: algoritmo de descenso por gradiente y regresión lineal (Diaps. 26–35):** Descenso guiado por el signo vs. descenso por gradiente $\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \eta^{(k)}\nabla_{\mathbf{x}}f(\mathbf{x}^{(k)})$, criterios de parada ($|f(\mathbf{x}^{(k+1)})-f(\mathbf{x}^{(k)})| < \varepsilon_1$, $\|\nabla f\| < \varepsilon_2$) y reproducción exacta del ejemplo de las **200 épocas en regresión lineal** (Diapositivas 34–35) con datos $\{(1,1),(2,3),(4,3)\}$, $\phi(x)=[1,x]$ y $\eta=0.1$ ($\mathbf{w}^{(0)}=[0,0] \to \mathbf{w}^{(1)}=[0.47, 1.27] \to \mathbf{w}^{(2)}=[0.25, 0.54] \to \mathbf{w}^{(200)}=[1.00, 0.57]$).
  5. **Tamaño de paso, los 4 regímenes, curvatura por dirección y convexidad (Diaps. 36–47):** Paso óptimo $\eta_{\text{opt}} = 1/a = (E'')^{-1}$ como inverso de la curvatura en cuadráticas, los 4 regímenes ($\eta < \eta_{\text{opt}}$ monótono, $\eta = \eta_{\text{opt}}$ un paso, $\eta_{\text{opt}} < \eta < 2\eta_{\text{opt}}$ oscila y converge, $\eta > 2\eta_{\text{opt}}$ diverge), verificación en Python sobre $E(w)=w^2$ con $\eta \in \{0.1, 0.5, 0.75, 1.1\}$ (Diaps. 44–45), trayectorias 2D sobre $E(\mathbf{w}) = \frac{1}{2}(w_1^2 + 3w_2^2)$ con cota $\eta < 2\min_i \eta_{i,\text{opt}}$ (Diaps. 41–43) y garantías de convexidad vs. inicialización en funciones no convexas (Diaps. 39–40).

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Integrar la **Unidad III: Descenso por Gradiente** con simuladores interactivos en la aplicación web y módulo de verificación matemática en Python.
* **Fases/Hitos principales:**
  1. Registro del plan de trabajo en `contexto/context8.md`.
  2. Creación del módulo en Python `descenso_gradiente.py` con las demostraciones analíticas y numéricas de las 47 diapositivas (incluyendo el sistema $3\times 3$ y Hessiano de Diaps. 24–25, las 200 épocas de regresión lineal de Diap. 35, las trayectorias 2D de Diap. 42 y el script de verificación de Diap. 45).
  3. Ampliación de la aplicación web `web/index.html` agregando la **Unidad Temática III: Descenso por Gradiente** con 4 laboratorios interactivos (Módulos 3.1 a 3.4, paneles 9 a 12) y autoevaluación interactiva III.
  4. Integración en `perceptron_xor.py`, `README.md` e `INFORME.md`, y ejecución de pruebas automatizadas (`--auto`).
* **Entregables:**
  - `contexto/context8.md`
  - `descenso_gradiente.py`
  - `web/index.html` (actualizado a 3 Unidades Académicas y 13 Laboratorios Interactivos)
  - `perceptron_xor.py`, `README.md` e `INFORME.md` actualizados

## 3. Referencias
* **Path del plan original:** `contexto/context8.md`
