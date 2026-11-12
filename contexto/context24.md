# Contexto 24: Integración de CD20 — Redes Neuronales Convolucionales (CNNs)
**Fecha de registro:** 2026-11-12

## 1. Instrucciones
* Integrar completamente el contenido de la cátedra **Redes Neuronales Convolucionales** (57 diapositivas) correspondiente a la **Unidad VIII de Fase 2 (CD20)** en el curso de Ciencia de Datos (Doctorado UCN).
* Fuentes académicas: Harvard CS109b (Pavlos Protopapas et al.).
* Ubicación curricular: Sucede a Regularización de Redes Neuronales (CD19 / Unidad VII) y precede a Redes Recurrentes (RNNs) para datos secuenciales.
* Requisitos de implementación técnica:
  1. **Módulo CLI en Python puro (`redes_convolucionales.py`):**
     - Sin dependencias de terceros (solo biblioteca estándar de Python 3).
     - Módulo 20.1: ¿Por qué las imágenes rompen a los MLP? (explosión de parámetros: CIFAR-10 $3\,073$, ImageNet 128 neuronas $\implies 19\,267\,712$ pesos; caso modesto $100\times 100 \implies 640\,064$; falta de invarianza a traslaciones; el caso del cisne y detectores manuales; localidad y jerarquía).
     - Módulo 20.2: Convolución, Filtros, Padding, Stride y Pooling (operación de suma ponderada local; micro-ejemplo exacto de $5 \times 5$ con borde vertical y filtro $3 \times 3$ dando salida con filas $[0, 9, 9]$; fórmula $O = \lfloor(W-K+2P)/S\rfloor + 1$ en los 3 casos de MNIST $28 \times 28$: $24\times 24$, $28\times 28$, $14\times 14$; no linealidad ReLU; max pooling y mean pooling exactos sobre lote $4 \times 4 \to 2 \times 2$: max $[7, 8; 3, 4]$ y mean $[3.5, 5.5; 1.75, 2.0]$).
     - Módulo 20.3: Armando una CNN y Conteo de Parámetros a Mano (las 3 capas: conv, pool, FC; conteo de 1 filtro $3\times 3\times 3 \implies 28$; 16 filtros sobre RGB $\implies 448$; red completa: Conv1 $224$, Conv2 $3\,216$, Flatten $4\,096$, FC1 $2\,097\,664$, FC2 $2\,052 \implies$ Total $2\,103\,156$ parámetros con $99.8\%$ en las capas densas; capa intermedia $14\times 14\times 16$ con 32 filtros $3\times 3 \implies 4\,640$ parámetros y salida $14\times 14\times 32$).
     - Módulo 20.4: Transfer Learning, Fine-Tuning y Tres Hitos (matriz $2\times 2$ de tareas/pesos; congelar base vs fine-tuning con tasas decrecientes; escenarios: 600 fotos de fauna altiplánica, 2M de fotos industriales, 400 espectrogramas sísmicos; hitos: LeNet-5 1998, AlexNet 2012, ResNet 2015 con atajos residuales $\mathbf{x} + F(\mathbf{x})$).
     - Menú interactivo con navegación y suite `--auto` que valide numéricamente todos los ejemplos.
  2. **Integración en Orquestador Maestro (`perceptron_xor.py`):**
     - Registrar CD20 en `UNIDADES_CATEDRA` como Unidad VIII (Num 18, ID 'cd20').
     - Incorporar opción en `menu_fase2()` y función `launch_cnn_module()`.
     - Agregar flags CLI `--cnn`, `--convolucional`, `--conv`, `-c20`, `--cd20`.
     - Incluir suite CD20 en la ejecución `--auto` global.
  3. **Suite Web Interactiva (`web/index.html`):**
     - Añadir Labs 69, 70, 71 y 72 (elevando el total de 69 a 73 laboratorios interactivos, paneles `#panel-69` a `#panel-72`).
     - Estilo visual de Unidad VIII en tonos Sky Blue / Electric Cyan (`#0ea5e9`, `#38bdf8`, `#0284c7`).
     - Visualizadores HTML5 Canvas con interactividad en tiempo real:
       * Lab 69: Explosión de Parámetros MLP vs. Localidad 2D e Invarianza por Traslación.
       * Lab 70: Convolución 2D Paso a Paso, Padding y Stride con Cálculo de Tamaño de Salida.
       * Lab 71: Simulador de Pooling (Max Pooling vs. Mean Pooling en Cuadrícula $4\times 4$).
       * Lab 72: Desglose Arquitectónico y Conteo de Parámetros de la CNN (Conv vs. FC 99.8%) + Autoevaluación VIII.
     - Quizzes 69 a 72 con retroalimentación inmediata.
     - Metadatos actualizados en `topicMetadata`, `ORDERED_UNITS` y `ORDERED_TABS`.
  4. **Documentación Teórica (`INFORME.md`):**
     - Añadir Sección 13 para Unidad VIII (CD20: Redes Neuronales Convolucionales) con deducciones matemáticas, tablas de conteo de parámetros, comparación MLP vs CNN y referencias históricas (LeCun et al. 1998, Krizhevsky et al. 2012, He et al. 2016).
     - Actualizar Conclusiones (Sección 14) y Referencias (Sección 15).
  5. **Documentación del Repositorio (`README.md`):**
     - Actualizar métricas globales: 18 Unidades Académicas • 1107 Diapositivas • 73 Laboratorios Interactivos.
     - Detallar CD20 en listas, desglose temático, comandos y estructura de archivos.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Implementar, integrar y verificar holísticamente la Unidad VIII de Fase 2 (CD20: Redes Neuronales Convolucionales) en los módulos CLI, orquestador principal, interfaz web interactiva y documentación formal.
* **Fases/Hitos principales:**
  1. *Fase 1 — Creación del Módulo CLI:* Codificar `redes_convolucionales.py` con los 4 módulos temáticos, menú interactivo y suite `--auto`.
  2. *Fase 2 — Integración en Orquestador:* Conectar CD20 en `perceptron_xor.py` (menú Fase 2, flags CLI, tests automáticos).
  3. *Fase 3 — Integración en Interfaz Web:* Extender `web/index.html` con 4 laboratorios (Labs 69 a 72), paneles visuales Canvas, controles interactivos, tabs y cuestionarios.
  4. *Fase 4 — Documentación Formal:* Redactar Sección 13 en `INFORME.md` y actualizar métricas/secciones en `README.md`.
  5. *Fase 5 — Verificación y Validación:* Ejecución de suites automáticas y comprobación sintáctica integral.
* **Entregables:**
  - `contexto/context24.md` (este documento).
  - `redes_convolucionales.py` (módulo CLI puro).
  - `perceptron_xor.py` (actualizado).
  - `web/index.html` (actualizado con Labs 69-72).
  - `INFORME.md` (Sección 13 incorporada).
  - `README.md` (métricas de 18 unidades y 73 laboratorios actualizadas).

## 3. Referencias
* **Path del plan original:** Documento PDF cátedra *Redes Neuronales Convolucionales* (Harvard CS109b / Ciencia de Datos UCN, 57 diapositivas).
