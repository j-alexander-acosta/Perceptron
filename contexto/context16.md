# Contexto 16 — Reorganización y Desaturación del Menú Principal y Navegación Interactiva

## 1. Solicitud del Usuario
- **Instrucción:** «Organiza el menú principal para que no se vea saturada y se puede navegar en la opción seleccionada.»
- **Diagnóstico del problema:**
  1. **Consola (`perceptron_xor.py`):** El menú principal imprimía de golpe 16 opciones con encabezados extensos por unidad en más de 35 líneas de texto saturando la terminal. Tras ejecutar una unidad, arrojaba al usuario de vuelta al menú raíz sin permitir una navegación secuencial guiada.
  2. **Web (`web/index.html`):** El selector de unidades desplegaba 10 tarjetas de gran tamaño en 4 filas ocupando más de 300px del alto de pantalla y empujando el contenido de los laboratorios fuera del primer plano visual.

---

## 2. Solución e Implementación en Consola (`perceptron_xor.py`)

### 2.1. Menú Maestro Jerárquico Desaturado (de 35+ líneas a solo 12 líneas limpias)
Se agruparon las opciones en 5 ejes temáticos y claros:
1. `[1] 📊 FASE 1: Fundamentos y Análisis de Datos (CD01 a CD04)`
2. `[2] 🧠 FASE 2: Redes Neuronales y Optimización (CD13 a CD18)`
3. `[3] ⚡ LABORATORIO CLÁSICO XOR: Perceptrón Multicapa (MLP 2 Capas)`
4. `[4] 🌐 INTERFACES GRÁFICAS: Suite Web (Navegador) & GUI de Escritorio (Tkinter)`
5. `[5] 🚀 VERIFICACIÓN COMPLETA AUTOMATIZADA (--auto, 10 Unidades)`
6. `[0] Salir del Sistema`

### 2.2. Submenús Enfocados por Fase
- **`menu_fase1()`:** Permite seleccionar entre CD01, CD02, CD03, CD04, correr todas las demos de Fase 1 o volver.
- **`menu_fase2()`:** Permite seleccionar entre CD13, CD14, CD15, CD16, CD17, CD18, correr todas las demos de Fase 2 o volver.
- **`menu_xor_lab()`:** Modo manual paso a paso, tabla de verdad y generación del gráfico .png.
- **`menu_interfaces()`:** Selección entre Suite Web y GUI Tkinter.

### 2.3. Sistema de Navegación Secuencial Encadenada (`ejecutar_unidad_con_navegacion`)
Al seleccionar cualquier unidad de cátedra, al finalizar su ejecución se presenta una barra interactiva de navegación:
```
─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─
 📍 Unidad actual: [CD01] (1 de 10)
   [S] Siguiente unidad (CD02)  |  [A] Anterior unidad  |  [R] Repetir (CD01)  |  [V] Volver al menú
─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─"─
```
- `S`: Avanza fluidamente a la siguiente unidad en el orden de cátedra (CD01 $\to$ CD02 $\to$ CD03 $\to$ CD04 $\to$ CD13 $\dots$).
- `A`: Retrocede a la unidad anterior.
- `R`: Repite la unidad seleccionada.
- `V`: Regresa al submenú enfocado.

---

## 3. Solución e Implementación en Suite Web (`web/index.html`)

### 3.1. Barra de Filtro de Fases (`.phase-tabs`)
- `[🌟 Todas las Unidades (10)]`
- `[📊 Fase 1: Fundamentos (CD01–CD04)]` $\implies$ Oculta las tarjetas de Fase 2 para despejar la vista.
- `[🧠 Fase 2: Redes Neuronales (CD13–CD18)]` $\implies$ Oculta las tarjetas de Fase 1.

### 3.2. Botón y Modo Compacto (`.unit-compact-bar`)
- Botón `[▲ Contraer Selector / ▼ Expandir Selector]` que repliega las 10 tarjetas a una barra delgada de 40px con la información de la unidad activa y controles de avance directo.

### 3.3. Botones de Navegación Rápida en el Banner
En la cabecera activa se agregaron 4 botones de acción inmediata:
- `[◀ Unidad Ant.]`: Salta a la unidad anterior según orden de cátedra.
- `[Unidad Sig. ▶]`: Salta a la siguiente unidad.
- `[◀ Lab Ant.]`: Salta al laboratorio previo dentro de los 41 laboratorios.
- `[Lab Sig. ▶]`: Salta al siguiente laboratorio.

---

## 4. Verificación y Resultados
- Verificación de sintaxis Python en `perceptron_xor.py`: Exitoso.
- Verificación no-interactiva de flujo de submenús y retorno (`printf "1\n1\n0\nv\nv\n0\n" | python3 perceptron_xor.py`): Exitoso.
- Verificación de sintaxis JavaScript en `web/index.html` con Node.js: Exitoso.
- Ejecución completa `--auto`: Todas las 10 unidades pasan al 100%.
