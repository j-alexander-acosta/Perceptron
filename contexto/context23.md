# Contexto 23: Integración de CD19 — Regularización de Redes Neuronales
**Fecha de registro:** 2026-11-12

## 1. Instrucciones
* Integrar completamente el contenido de la cátedra **CD19: Regularización de Redes Neuronales** (46 diapositivas) impartida en el marco del Doctorado UCN (Ciencia de Datos).
* Fuentes académicas: CMU 11-785 (Bhiksha Raj, Rita Singh et al.) y Stanford CS221 (Percy Liang et al.).
* Ubicación curricular: Corresponde a la **Unidad VII de Fase 2 (Redes Neuronales)**, sucediendo a Descenso por Gradiente Estocástico y Lotes (CD18) y precediendo a Arquitecturas Especializadas (CNNs y RNNs).
* Requisitos de implementación técnica:
  1. **Módulo CLI en Python puro (`regularizacion_redes.py`):**
     - Sin dependencias de terceros (solo biblioteca estándar de Python 3).
     - Módulo 19.1: Divergencias (L2 vs KL), gradiente de error $\nabla_z \text{div} = y - d$, saturación sigmoide ($\sigma'(6) \approx 0.0025$), subespecificación de datos ($2^{100} \approx 10^{30}$) y suavidad del perceptrón ($w=0.5$ vs $w=5$).
     - Módulo 19.2: Regularización L2 y Weight Decay (objetivo $\min_W \frac{1}{T}\sum_t \text{div} + \frac{1}{2}\lambda \sum \|W_k\|_F^2$, actualización $(1-\eta\lambda)W$, ejemplo numérico $\eta=0.1, \lambda=0.1 \implies 0.99$, y efecto de $\lambda$).
     - Módulo 19.3: Dropout y Dropout Invertido (sorteo Bernoulli con probabilidad de retención $\alpha$, $2^N$ subredes compartiendo pesos, escalamiento en prueba $f(z)=2, \alpha=0.8 \to 1.6$, dropout invertido en entrenamiento $2/0.8 = 2.5$, DropConnect, Zoneout).
     - Módulo 19.4: Trucos del Oficio y Normalización por Lotes (Early Stopping, Grokking, Gradient Clipping $\theta=5$, Data Augmentation, y BatchNorm Ioffe-Szegedy 2015 con micro-ejemplos exactos $\{1,2,3,6\}$ y $\{2,4,6,8\}$, medias móviles y factor de Bessel $B/(B-1)$).
     - Menú interactivo con navegación completa y modo `--auto` que evalúe y verifique con aserciones todos los ejemplos numéricos de la cátedra.
  2. **Integración en Orquestador Maestro (`perceptron_xor.py`):**
     - Registrar CD19 en `UNIDADES_CATEDRA` como Unidad VII (Num 19, id 'cd19').
     - Incorporar opción en `menu_fase2()` y función `launch_regularization_module()`.
     - Agregar flags CLI `--regularization`, `--reg`, `--dropout`, `--batchnorm`, `-c19`, `--cd19`.
     - Incluir suite CD19 en el pipeline `--auto` global.
  3. **Suite Web Interactiva (`web/index.html`):**
     - Añadir Labs 65, 66, 67 y 68 (elevando el total de 65 a 69 laboratorios interactivos, paneles `#panel-65` a `#panel-68`).
     - Estilo visual de Unidad VII en tonos Rose / Coral / Pink (`#f43f5e`, `#fb7185`, `#e11d48`).
     - Visualizadores HTML5 Canvas con interactividad en tiempo real:
       * Lab 65: Paisaje de Divergencias y Suavidad de Activación (L2 vs KL con saturación sigmoide y pendiente $|w|$).
       * Lab 66: Decaimiento de Pesos ($L_2$ Regularization / Weight Decay con factor $1-\eta\lambda$ y contracción de esferas).
       * Lab 67: Simulador de Dropout y Dropout Invertido (subredes activas con tasa $\alpha$, escalamiento dinámico).
       * Lab 68: Batch Normalization interactiva (cálculo de $\mu_B, \sigma_B^2$, estandarización $u_i$, escalamiento $\gamma$ y sesgo $\beta$).
     - Quizzes 65 a 68 rigurosos con retroalimentación inmediata.
     - Metadatos actualizados en `topicMetadata`, `ORDERED_UNITS` y `ORDERED_TABS`.
  4. **Documentación Teórica (`INFORME.md`):**
     - Agregar Sección 12 completa para Unidad VII (CD19) con deducciones matemáticas rigurosas, tablas comparativas, código PyTorch y citas bibliográficas (Srivastava et al. 2014, Ioffe & Szegedy 2015, Power et al. 2022, Pascanu et al. 2013).
  5. **Documentación del Repositorio (`README.md`):**
     - Actualizar métricas globales: 17 Unidades Académicas • 1050 Diapositivas • 69 Laboratorios Interactivos.
     - Describir CD19, comandos CLI y estructura de archivos.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar, integrar y verificar holísticamente la Unidad VII de Fase 2 (CD19: Regularización de Redes Neuronales) en los módulos CLI, orquestador principal, interfaz web interactiva y documentación técnica.
* **Fases/Hitos principales:**
  1. *Fase 1 — Creación del Módulo CLI:* Codificar `regularizacion_redes.py` con los 4 módulos temáticos, menú interactivo y suite `--auto`.
  2. *Fase 2 — Integración en Orquestador:* Conectar CD19 en `perceptron_xor.py` (menú de Fase 2, flags CLI, tests automáticos).
  3. *Fase 3 — Integración en Interfaz Web:* Extender `web/index.html` con 4 laboratorios (Labs 65 a 68), paneles visuales Canvas, controles interactivos, tabs y cuestionarios.
  4. *Fase 4 — Documentación Formal:* Redactar Sección 12 en `INFORME.md` y actualizar métricas/secciones en `README.md`.
  5. *Fase 5 — Verificación y Validación:* Ejecución de suites automáticas y comprobación sintáctica integral.
* **Entregables:**
  - `contexto/context23.md` (este documento).
  - `regularizacion_redes.py` (módulo CLI puro).
  - `perceptron_xor.py` (actualizado).
  - `web/index.html` (actualizado con Labs 65-68).
  - `INFORME.md` (Sección 12 incorporada).
  - `README.md` (métricas de 17 unidades y 69 laboratorios actualizadas).

## 3. Referencias
* **Path del plan original:** `file:///Users/alexanderacosta/.gemini/antigravity/brain/507343f4-6cd6-4373-a3da-2e65def7a36d/.user_uploaded/media_1794487951588_acc00693.pdf`
