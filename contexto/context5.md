# Contexto 5: Creación del Entorno Gráfico Interactivo para el Perceptrón XOR
**Fecha de registro:** 2026-10-26

## 1. Instrucciones
* Desarrollar un entorno gráfico amigable, intuitivo y visualmente atractivo para interactuar con el Perceptrón Multicapa XOR.
* Proveer dos alternativas integradas:
  1. **Aplicación Web Interactiva (`web/index.html`):** Visualizador dinámico con diagrama de la red neuronal animado en SVG/Canvas, activación en tiempo real de neuronas (brillo, pesos y sumas ponderadas), gráfico 2D del espacio cartesiano interactivo y tabla de verdad sincronizada.
  2. **Interfaz Gráfica de Escritorio (`gui_xor.py`):** Aplicación nativa en Python con `tkinter` que se ejecuta directamente desde la terminal con `python3 gui_xor.py` o desde el menú de `perceptron_xor.py`.
* Permitir cambiar interactivamente las entradas $X$ e $Y$ (botones/selectores conmutables $0 \leftrightarrow 1$) y observar de inmediato el flujo de la señal a través de la capa oculta y la capa de salida.
* Actualizar `README.md` e `INFORME.md` para reflejar el nuevo entorno gráfico.
* Confirmar y sincronizar los cambios con el repositorio GitHub.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Construir una interfaz gráfica moderna e interactiva que haga accesible y didáctica la comprensión del Perceptrón Multicapa para XOR.
* **Fases/Hitos principales:**
  1. Implementación de la aplicación de escritorio nativa `gui_xor.py` con Tkinter.
  2. Implementación de la aplicación web interactiva `web/` con estética premium, animaciones de flujo y visualización gráfica.
  3. Integración en `perceptron_xor.py` como opción de lanzamiento gráfico.
  4. Sincronización en Git con el repositorio remoto.
* **Entregables:**
  - `gui_xor.py`
  - `web/index.html`
  - Archivos actualizados (`README.md`, `INFORME.md`, `perceptron_xor.py`, `contexto/context5.md`).

## 3. Referencias
* **Path del plan original:** `contexto/context5.md`
