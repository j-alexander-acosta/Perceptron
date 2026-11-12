# Contexto 26: Integración Curricular de CD12 - Redes Neuronales y Conexionismo
**Fecha de registro:** 2026-11-12

## 1. Instrucciones
* **Origen de Cátedra:** Presentación oficial de 60 diapositivas titulada *«Redes Neuronales y Conexionismo»*, correspondiente al Doctorado en Informática de la Universidad Católica del Norte (UCN), basada en material de Stanford CS221 (Percy Liang et al.) y CMU 11-785 (Bhiksha Raj et al.).
* **Secuencia Curricular y Ubicación:** Llena de forma natural y secuencial el espacio reservado entre `CD11` (Aspectos Prácticos y Evaluación) y `CD13` (Aproximación Universal en Redes Neuronales), siendo designada como **CD12: Redes Neuronales y Conexionismo (60 Diapositivas)** y actuando como el puente fundacional e histórico que conecta la Ciencia de Datos / Machine Learning clásico (Fase 1) con el Aprendizaje Profundo / Redes Neuronales (Fase 2).
* **Métricas Globales de la Suite:**
  - Unidades Académicas: Pasa de 19 a **20 Unidades Académicas** (12 en Fase 1: `CD01` a `CD12`; 8 en Fase 2: `CD13` a `CD20`).
  - Total de Diapositivas: Pasa de 1.200 a **1.260 Diapositivas**.
  - Laboratorios Interactivos: Pasa de 77 a **81 Laboratorios Interactivos** (4 nuevos laboratorios: 12.1 a 12.4, integrados en la consola y en los paneles `#panel-77` a `#panel-80` de la suite web).
* **Requisitos Técnicos:**
  - Python 3 estándar (sin dependencias externas obligatorias para la ejecución de la suite).
  - Reproducción numérica y demostrativa 100% fiel de las 60 diapositivas (código de aprendizaje OR en NumPy/Python puro, estimaciones de Bain de 1873 vs. neurociencia actual, divergencia desbocada de Hebb vs. estabilización de Oja, imposibilidad de XOR en neurona única y solución con MLP de 3 neuronas, clasificador de dos triángulos disjuntos con 9 neuronas, y aproximación armónica $\sin(2x + \pi/2) \equiv \cos(2x)$ con 1 neurona).
  - Interfaz web interactiva en `web/index.html` con paleta visual distintiva (Violeta / Púrpura Conexionista), paneles `#panel-77` a `#panel-80`, visualizadores en HTML5 Canvas y banco de preguntas `quizData` para los laboratorios 77 a 80.
  - Documentación exhaustiva en `INFORME.md` (Sección 0.15) y actualización métrica en `README.md`.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Desarrollar e integrar plenamente la unidad `CD12: Redes Neuronales y Conexionismo` en la suite doctoral de Ciencia de Datos mediante CLI interactivo, lanzador central, suite web en navegador e informe académico riguroso.
* **Fases/Hitos principales:**
  1. **Módulo 12.1 — Asociacionismo y Conexionismo (Diaps. 1–21):**
     * Paradigma cognitivo: Asociacionismo (Aristóteles, Locke, Hume, Pavlov) y las 4 leyes de asociación (Contigüidad, Frecuencia, Similitud, Contraste).
     * El amanecer del conexionismo: David Hartley (1749, *vibratiúnculas*).
     * Tesis fundamental de Alexander Bain (1873, *Mind and Body*): *"La información no está en las unidades: está en las conexiones entre ellas"*.
     * Agrupaciones neuronales, anticipación del aprendizaje hebbiano (1873 vs. 1949) y la paradoja de la escala: estimación de Bain ($5 \times 10^9$ conexiones) vs. cerebro humano real ($8.6 \times 10^{10}$ neuronas y $10^{14}$ sinapsis, factor $\times 20\,000$).
     * Máquinas conexionistas vs. arquitectura Von Neumann (programa y datos separados vs. la red es el programa y la memoria).
     * Máquinas conexionistas de Turing (1948): Tipo A (redes aleatorias desorganizadas de compuertas NAND) y Tipo B (redes entrenables con modificadores adaptativos).
     * Los 8 ingredientes de un sistema PDP (Rumelhart, Hinton, McClelland, 1986).
  2. **Módulo 12.2 — La Neurona Artificial: McCulloch-Pitts (1943) y Regla de Hebb (1949) (Diaps. 22–30):**
     * Biología básica: soma, dendritas (múltiples entradas) y axón (salida única hacia múltiples destinos).
     * Modelo de McCulloch & Pitts (1943): lógica proposicional con sinapsis excitatorias y sinapsis inhibitorias absolutas. Implementación de compuertas booleanas AND ($\theta=2$), OR ($\theta=1$), NOT ($\theta=0$, peso $-1$). Redes con ciclos para memoria.
     * La carencia fundamental de McCulloch-Pitts: ausencia de regla de aprendizaje automático.
     * Donald Hebb (1949): *"Neuronas que se disparan juntas, se conectan"* ($w_{xy} \leftarrow w_{xy} + \eta x y$).
     * Diagnóstico de inestabilidad: ausencia de competencia y de cota superior ($x=y=1, \eta=0.1 \implies w \to \infty$).
     * Regulación y estabilidad: Regla de Oja (1982) ($\Delta w = \eta y (x - y w)$) y GHA de Sanger (1989).
  3. **Módulo 12.3 — El Perceptrón de Rosenblatt, Límites de XOR y Perceptrón Multicapa (Diaps. 31–41):**
     * Frank Rosenblatt (1958): retina sensorial ($S$), celdas de asociación ($A$, pesos fijos) y respuesta ($R$, pesos aprendibles).
     * Modelo de umbral: $y = \mathbb{I}(\sum w_i x_i - T \ge 0)$. Regla de aprendizaje: $w \leftarrow w + \eta (d(x) - y(x)) x$. Teorema de convergencia para datos linealmente separables.
     * Código de aprendizaje en Python puro de compuerta OR extendida ($w = [-1.0, 1.0, 1.0]$).
     * Titulares del New York Times (1958) y la sobre-promesa.
     * Demostración analítica de la imposibilidad de resolver XOR con una sola neurona: sistema incompatible de 4 desigualdades ($w_1 < T, w_2 < T, w_1 + w_2 \ge T, 0 < T$).
     * Minsky & Papert (1969, *Perceptrons*) y el primer invierno de la IA.
     * Solución con Perceptrón Multicapa (MLP): $X \oplus Y = (X \lor Y) \land (\bar{X} \lor \bar{Y})$. Arquitectura de 2 neuronas ocultas + 1 salida.
  4. **Módulo 12.4 — De lo Booleano a los Reales, Universalidad e Hitos Históricos (Diaps. 42–60):**
     * Entradas reales $x_i \in \mathbb{R}$, combinación afín $\sum w_i x_i + b$, y activaciones suaves (Escalón vs. Sigmoide $\sigma(z)$ vs. $\text{ReLU}(z)$).
     * Clasificador Universal: de un polígono convexo ($n$ lados $\implies n$ neuronas + 1 AND) a dos triángulos disjuntos (región no conexa: 6 semiplanos + 2 AND + 1 OR $\implies \mathbf{9\text{ neuronas}}$ en 2 capas ocultas).
     * Regresor Universal: síntesis de pulsos rectangulares $f(x) = \text{step}(x-T_1) - \text{step}(x-T_2)$ y suma de Riemann para cualquier función continua.
     * Teorema del tamaño vs. activación: para $y = \cos(2x)$, una función de activación sinusoidal $\sin(wx + b)$ requiere **1 sola neurona** ($\sin(2x + \pi/2) \equiv \cos(2x)$).
     * Por qué se estancó la optimización de MLPs con escalón: error binario plano a trozos ($\nabla = \mathbf{0}$) y explosión combinatoria NP-difícil.
     * Cronología del conexionismo y deep learning: Neocognitrón (1980), Backpropagation (1986), LeNet (1989), Deep Learning (2006), Reconocimiento de voz (2009), AlexNet (2012), AlphaGo (2016). IA Simbólica vs. IA Neuronal.

## 3. Archivo Original
* **Ruta del documento analizado:** `/Users/alexanderacosta/.gemini/antigravity/brain/507343f4-6cd6-4373-a3da-2e65def7a36d/.user_uploaded/media_1794503986877_bcfaef1b.pdf`
* **Nombre de archivo cargado:** `CD12-Redes-Neuronales-y-Conexionismo.pdf`
