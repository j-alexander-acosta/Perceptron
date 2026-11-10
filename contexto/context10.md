# Contexto 10: Integración de la Unidad V — Problemas de Convergencia en el Entrenamiento (58 Diapositivas)

**Fecha de registro:** 2026-11-09

## 1. Instrucciones
- Incorporar la nueva presentación académica de 58 diapositivas (**«Problemas de Convergencia en el Entrenamiento»**, Ciencia de Datos — UCN, basado en material de CMU 11-785 de B. Raj et al.) como la **Unidad Temática V** del proyecto.
- Desarrollar el módulo computacional interactivo y autoverificable en Python puro (`problemas_convergencia.py`) que implemente y verifique numéricamente las 5 secciones del material:
  1. **¿Encuentra *backprop* la solución correcta? (Diaps. 7–16):** La divergencia diferenciable como sustituto (*proxy*) del error de clasificación $0\text{-}1$; el experimento clásico de **Brady, Raghavan y Slawny (1989)** con 3 puntos linealmente separables $(1,0)\to +1$, $(0,1)\to +1$, $(-1,0)\to -1$ y margen $\varepsilon$ ($w_1=w_2=u=\sigma^{-1}(1-\varepsilon), b=0$); la adición del cuarto punto «saboteador» (*spoiler*) lejano $(0, -t)\to +1$; demostración de la saturación exponencial de la derivada logística $\sigma'(z_4)\sim e^{-ut}$ frente al crecimiento lineal de $t$ en $\frac{d\,\text{div}_4}{dw_2} = 2(1-\varepsilon-\sigma(z_4))\sigma'(z_4)t \to 0$, creando un mínimo local ancho que deja $(0,-t)$ mal clasificado mientras el perceptrón separa los 4 puntos; y la compensación sesgo-varianza.
  2. **La superficie de pérdida (Diaps. 17–20):** Paisajes no convexos, mínimos locales, mesetas, **puntos de silla** ($E = \frac{1}{2}(w_1^2 - w_2^2)$ con valores propios del Hessiano de signos opuestos) y la evidencia teórica y empírica de **Baldi y Hornik (1989)**, **Dauphin et al. (2014)**, **Choromanska et al. (2015)** y **Swirszcz et al. (2016)**.
  3. **Convergencia, tasa de aprendizaje y número de condición (Diaps. 21–33):** El lente convexo y el *efecto farol*; tasa de convergencia lineal $R = \frac{|E(\mathbf{w}^{(k+1)})-E(\mathbf{w}^*)|}{|E(\mathbf{w}^{(k)})-E(\mathbf{w}^*)|}$; paso óptimo $\eta_{i,\text{opt}} = 1/a_i$ y umbral de estabilidad global $\eta < 2\min_i \eta_{i,\text{opt}}$; reproducción exacta de las **6 trayectorias de la Diapositiva 27** sobre $E = \frac{1}{2}(w_1^2 + 3w_2^2)$ desde $\mathbf{w}^{(0)} = (-8, 4)$ con $\eta \in \{0.25, 1/3, 0.5, 2/3, 0.75, 1.0\}$; el **número de condición** $\kappa = a_{\max}/a_{\min}$ y factor óptimo $\frac{\kappa-1}{\kappa+1}$; y comparación entre funciones fuertemente convexas $O(\log(1/\delta))$ y funciones convexas solo Lipschitz-suaves $O(1/\delta)$.
  4. **Métodos de segundo orden: normalizar la curvatura (Diaps. 34–44):** Escalamiento de ejes $\hat{\mathbf{w}} = A^{1/2}\mathbf{w}$ ($A = U\Lambda U^\top$), actualización normalizada $\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \eta A^{-1}\nabla_{\mathbf{w}}E(\mathbf{w}^{(k)})$, matriz **Hessiana** $H_E(\mathbf{w})$ y **método de Newton iterado**; demostración de los **2 problemas de Newton** (explosión cuadrática $N^2 = 10^{10}$ entradas para $N=100\,000$ e inversión $O(N^3)$, y atracción hacia **puntos de silla** por curvatura negativa) y métodos cuasi-Newton (**BFGS / L-BFGS** y **Levenberg-Marquardt** $H \approx J^\top J + \lambda I$).
  5. **Tasas decrecientes y pasos adaptativos por coordenada: RProp y QuickProp (Diaps. 45–58):** Exploración inicial con $\eta$ grande para escapar de mínimos locales mediocres seguida de esquemas de decaimiento ($\eta_0/k$, $\eta_0/k^2$, $\eta_0 e^{-\beta(k-1)}$, escalones); algoritmo **RProp** (*Resilient Propagation*, usando solo el signo de la derivada parcial con factores $\alpha > 1$, $\beta < 1$ y retroceso de paso cuando cambia el signo); algoritmo **QuickProp** ($\Delta w^{(k)} = \Delta w^{(k-1)}\frac{E'(w^{(k)})}{E'(w^{(k-1)}) - E'(w^{(k)})}$ como ajuste parabólico local por diferencia finita de derivadas); y motivación hacia **Momentum**.
- Expandir la plataforma interactiva `web/index.html` para incorporar la **Unidad V: Problemas de Convergencia en el Entrenamiento** con 4 nuevos laboratorios interactivos (`panel-17` a `panel-20`, alcanzando **5 Unidades y 21 Laboratorios Interactivos**) y ampliar el banco de autoevaluación a 20 preguntas.
- Integrar el nuevo módulo en `perceptron_xor.py` (`--convergencia` / `-c`, menú interactivo `[8]` y suite `--auto`), `README.md` e `INFORME.md`.

## 2. Plan de Trabajo (Resumen)
1. **Registro de Contexto (`contexto/context10.md`):** Documentar las instrucciones y el plan de trabajo estructurado siguiendo el estándar del repositorio.
2. **Módulo Computacional (`problemas_convergencia.py`):**
   - Implementar las 4 demostraciones ejecutables y verificables (`demo_brady_saboteadores`, `demo_superficie_y_numero_condicion`, `demo_newton_y_hessiano`, `demo_tasas_rprop_quickprop`) con soporte para modo interactivo y bandera `--auto`.
3. **Expansión de la Plataforma Web (`web/index.html`):**
   - Agregar el selector de **Unidad V: Problemas de Convergencia (58 Diaps.)** y su barra de pestañas (`#unit5Tabs`) con los módulos 5.1 a 5.4 (`panel-17` a `panel-20`):
     - **Módulo 5.1 (`panel-17`):** Simulador del experimento de Brady, Raghavan y Slawny (1989) con el cuarto punto saboteador $(0, -t)$, frontera de *Backprop* vs. Perceptrón y gráfica de apagado exponencial del gradiente por saturación de la sigmoide.
     - **Módulo 5.2 (`panel-18`):** Explorador de las 6 trayectorias exactas sobre $E = \frac{1}{2}(w_1^2 + 3w_2^2)$ desde $(-8,4)$, número de condición $\kappa$, punto de silla 2D/3D y comparador de tasas $O(\log(1/\delta))$ vs. $O(1/\delta)$.
     - **Módulo 5.3 (`panel-19`):** Laboratorio de segundo orden: transformación de coordenadas $\hat{\mathbf{w}} = A^{1/2}\mathbf{w}$, método de Newton vs. Descenso por Gradiente vs. Levenberg-Marquardt tanto en cuencos anisotrópicos rotados como en la trampa de curvatura negativa de un punto de silla.
     - **Módulo 5.4 (`panel-20`):** Simulador de escape de mínimos locales con tasas decrecientes, comparador paso a paso de **GD vs. RProp vs. QuickProp** y **Autoevaluación Interactiva V** (preguntas 17 a 20).
4. **Integración y Documentación (`perceptron_xor.py`, `README.md`, `INFORME.md`):**
   - Conectar `problemas_convergencia.py` al menú principal y a `--auto`, actualizar la documentación técnica y validar todas las pruebas automatizadas.

## 3. Referencias
- Presentación adjunta: `CD17-Problemas-de-Convergencia.pdf` (*Problemas de Convergencia en el Entrenamiento*, Ciencia de Datos — UCN, 58 diapositivas, basado en material de CMU 11-785 de B. Raj et al.).
- Artículos fundamentales citados en la cátedra:
  - Brady, M. L., Raghavan, R., & Slawny, J. (1989). *Back propagation fails to separate where perceptrons succeed*. IEEE Transactions on Circuits and Systems, 36(5), 665–674.
  - Baldi, P., & Hornik, K. (1989). *Neural networks and principal component analysis: Learning from examples without local minima*. Neural Networks, 2(1), 53–58.
  - Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y. (2014). *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization*. NIPS 2014.
  - Choromanska, A., Henaff, M., Mathieu, M., Ben Arous, G., & LeCun, Y. (2015). *The loss surfaces of multilayer networks*. AISTATS 2015.
  - Swirszcz, G., Czarnecki, W. M., & Pascanu, R. (2016). *Local minima in training of neural networks*. arXiv:1611.06310.
  - Riedmiller, M., & Braun, H. (1993). *A direct adaptive method for faster backpropagation learning: The RPROP algorithm*. IEEE ICNN.
  - Fahlman, S. E. (1988). *An empirical study of learning speed in back-propagation networks* (QuickProp). CMU-CS-88-162.
- Contextos previos del proyecto: `contexto/context8.md` y `contexto/context9.md`.
