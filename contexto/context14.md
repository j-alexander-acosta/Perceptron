# Contexto 14: Integración de la Unidad CD03 (Tercera en Orden) — Visualización de Datos Efectiva y Engañosa (103 Diapositivas)
**Fecha de registro:** 2026-11-09

## 1. Instrucciones
* Incorporar la nueva presentación académica de 103 diapositivas (**«CD03: Visualización de Datos Efectiva y Engañosa»**, Ciencia de Datos — UCN, basado en material de CMU 15-388/688 de Z. Kolter y la sección de visualización engañosa adaptada de Michael Correll, Tableau Research) en la **tercera posición** del proyecto (inmediatamente después de `CD01` y `CD02`, y antes de `CD13`–`CD18`), respetando estrictamente el orden numérico del curso (`CD01`, `CD02`, `CD03`, ..., `CD13`–`CD18`).
* Desarrollar el módulo computacional interactivo y autoverificable en Python puro (`visualizacion_datos.py`) que implemente y verifique numérica y conceptualmente las 4 secciones de la unidad `CD03`:
  1. **Por Qué Visualizar, Datasaurus Dozen, Escalas de Medición y Catálogo de Gráficos (Diaps. 1–27 y 50–56):**
     * Ubicación en el proceso de 5 pasos: la visualización trabaja en **dos pasos a la vez** (**explorar** los datos propios y **comunicar** resultados a otros, Diap. 4). Contraste entre **Explorar** (para ti, rápido y desechable) y **Presentar** (para otros, pulido y con un mensaje, Diap. 6), y máxima de John Tukey (Diap. 7).
     * **Cuarteto de Anscombe (1973, Diaps. 8–9) y Datasaurus Dozen (Matejka y Fitzmaurice, CHI 2017, Diaps. 10–12):** 13 nubes de $n = 142$ puntos (*dino*, *x shape*, *star*, *circle*, *slant up*, *away*, etc.) con las 5 mismas estadísticas congeladas hasta el segundo decimal ($\bar{x} = 54.26$, $\bar{y} = 47.83$, $s_x = 16.76$, $s_y = 26.93$, $r = -0.06$), explicadas por la propiedad matemática $n > m$ grados de libertad.
     * **Las 4 escalas de medición (Stevens, Diaps. 13–14):**
       - *Nominal:* categorías sin orden (`=`, $\neq$; ej. `club del jugador`, mascota).
       - *Ordinal:* categorías con orden (`=`, $\neq$, `<`, `>`; ej. `puesto en el ranking`, estrellas 1–5; la distancia no significa nada).
       - *Intervalo:* numérico con cero arbitrario (`+`, `-`; ej. temperatura en °C, `año de debut`; «20 °C no es el doble de 10 °C»).
       - *Razón:* numérico con cero absoluto que significa «nada» (todas, incluida `÷`; ej. Kelvin, masa, `precio de mercado`).
     * **Catálogo por dimensionalidad, Caja y Bigotes a mano, Dispersión 3D y Ancho de Intervalo (Diaps. 15–27, 50–56):**
       - 1D (barras, circular, histograma), 2D (dispersión, líneas, caja y bigotes, mapa de calor: matriz vs. densidad 2D) y 3D+ (matriz de dispersión, burbujas, dispersión con color).
       - **Cálculo exacto del Box Plot (Diap. 21):** Para $\{2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 15\}$ ($n=11$), $\text{mediana} = 5.0$, $Q_1 = 3.5$, $Q_3 = 7.5$, $\text{IQR} = Q_3 - Q_1 = 4.0$, cerca superior $Q_3 + 1.5\,\text{IQR} = \mathbf{13.5}$, quedando el $\mathbf{15}$ marcado como valor atípico (*outlier*).
       - **Por qué no usar dispersión 3D con perspectiva (Diaps. 24–25):** (1) El ángulo de cámara elige la conclusión (240 puntos se ven como 1 grupo desde la cámara A y 2 grupos desde la cámara B) y (2) oclusión: de un grupo real de 30 observaciones rojas detrás de la nube sobrevive solo el **$22\%$** de su tinta.
       - **El árbol de Abela (2009), Barras vs. Líneas (Zacks y Tversky, 1999) y el ancho de intervalo en histogramas (Diaps. 50–56):** En el conjunto bimodal de 200 datos (Diap. 56), *bins* de ancho $1.0$ muestran 3 barras planas («distribución uniforme»), mientras que *bins* de ancho $0.2$ revelan los **dos modos** (dos poblaciones mezcladas).
  2. **Caso Clásico: Los Antibióticos de Will Burtin (1951) y el Gráfico que Corrigió a la Biología (Diaps. 28–38):**
     * **Reconstrucción 100% exacta de la tabla de Burtin (1951, *Scope*; Wainer y Lysen, 2009, Diap. 29):** 16 bacterias (7 Gram positivas y 9 Gram negativas) frente a 3 antibióticos (*penicilina*, *estreptomicina* y *neomicina*), midiendo la **CIM** (concentración inhibitoria mínima, de $0.001$ a $870$, 6 órdenes de magnitud, donde **menor CIM = más potente**).
     * **Pregunta 1 (del médico, Diaps. 30–33):** Medianas exactas de CIM por grupo (Diap. 33):
       - *Gram positivas (7):* `penicilina = 0.005`, `estreptomicina = 1.0`, `neomicina = 0.1` (la penicilina es **$200$ veces** más potente que la estreptomicina).
       - *Gram negativas (9):* `penicilina = 100.0`, `estreptomicina = 1.0`, `neomicina = 0.1` (la penicilina pasa a ser **$20\,000$ veces** menos potente que en Gram positivas, y gana la neomicina).
     * **Pregunta 2 (del biólogo, Diaps. 34–36):** Al graficar penicilina vs. neomicina en escala logarítmica coloreando por género, cada género se agrupa y saltan dos anomalías («?») que anticiparon en décadas a la microbiología:
       - `Diplococcus pneumoniae` ($\text{pen}=0.005, \text{neo}=10$) cae junto a *S. hemolyticus* y *S. viridans* $\to$ reclasificada en **1974** como *Streptococcus pneumoniae*.
       - `Streptococcus fecalis` ($\text{pen}=1.0, \text{neo}=0.1$) cae de $200$ a $1\,000$ veces lejos de los estreptococos $\to$ reclasificada en **1984** en el nuevo género *Enterococcus faecalis*.
     * **Recetar leyendo el gráfico (Diap. 37):** *Brucella abortus* ($\text{pen}=1, \text{neo}=0.02 \to$ **neomicina**, $50$ veces menor CIM); *Bacillus anthracis* ($\text{pen}=0.001, \text{neo}=0.007 \to$ **penicilina**); y *S. fecalis* (donde recetar penicilina «por llamarse estreptococo» era un error de $200\times$ a $1\,000\times$).
  3. **Integridad Gráfica (Tufte, 1983), Lie Factor, Data-Ink Ratio, Percepción (Stevens, Cleveland-McGill, Mackinlay) y Color (Diaps. 39–49 y 57–75):**
     * **Los 5 principios de visualización efectiva y el *Lie Factor* de Tufte (1983, Diaps. 39–41):**
       $$\text{LieFactor} = \frac{\text{tamaño del efecto en el gráfico}}{\text{tamaño del efecto en los datos}}$$
       **Cálculo exacto a mano del caso *Bush Tax Cuts* (Fox Business, 2012, Diap. 41):** Tasa hoy $= 35\%$, tras el alza $= 39.6\%$, con eje truncado en $34\%$. Razón visual de alturas $= \frac{39.6 - 34}{35 - 34} = \mathbf{5.6}$ veces más alta ($\text{efecto en el gráfico} = 5.6 - 1 = \mathbf{4.6} = +460\%$), mientras que el alza real en los datos es $\frac{39.6 - 35}{35} = \frac{4.6}{35} \approx \mathbf{0.1314} \approx \mathbf{13.1\%}$, dando $\text{LieFactor} = \frac{4.6}{0.1314} \approx \mathbf{35.0}$.
     * **Eje del tiempo desigual (Diap. 42), el dilema del cero (Diap. 43: obligatorio en barras, opcional declarado en líneas), *Data-Ink Ratio*, *Chartjunk*, circulares 3D (Kevin Fox, 2010) y la auditoría de los 4 problemas de las marcas A, B, C, D (Diaps. 44–49):** En las marcas $A=52\%, B=53\%, C=54\%, D=56\%$ con eje truncado en $50\%$, $D$ ($56-50=6$) parece el **triple** de $A$ ($52-50=2$).
     * **Efectividad perceptual: Ley de Potencias de Stevens (1961), Cleveland y McGill (1984) y Mackinlay (1986) (Diaps. 57–66):**
       - $\text{percibido} \propto \text{intensidad}^\beta$: Longitud ($\beta = 1.0 \implies 10^{1.0} = \mathbf{10.0\times}$, fiel), Área ($\beta = 0.7 \implies 10^{0.7} \approx \mathbf{5.0\times}$, a la mitad; radios en razón $\sqrt{10}\approx 3.16$), Brillo ($\beta = 0.33 \implies 10^{0.33} \approx \mathbf{2.1\times}$, a un quinto; $2\times$ en oscuridad $30\% \to 60\%$ se percibe como $2^{0.5}\approx \mathbf{1.4\times}$), y descarga eléctrica ($\beta = 3.5 \implies 2^{3.5} \approx \mathbf{11.3\times} \approx 11$ veces peor).
       - Los 6 niveles de **Cleveland y McGill (1984)** (1. Posición en escala común $>$ 2. Posición no alineada $>$ 3. Longitud, dirección, ángulo $>$ 4. Área $>$ 5. Volumen, curvatura $>$ 6. Sombreado, saturación) y las 3 columnas de **Mackinlay (1986)** (donde la posición gana en las 3 y el **tono** sube al 2.º lugar para datos nominales).
     * **Uso estratégico del color (Diaps. 67–74):** Regla de Ware (máximo $5$ a $8$ colores categóricos), por qué el arcoíris (`jet`) engaña (su luminancia Rec. 709 sube y baja 3 veces creando bordes falsos, mientras `viridis` crece monótonamente de $\mathbf{0.02}$ a $\mathbf{0.78}$, Diaps. 68–70), daltonismo (~$8\%$ de los hombres tiene deficiencia rojo/verde; **azul vs. naranjo** es la pareja segura por defecto, Machado et al., 2009, Diap. 72) y ColorBrewer.
  4. **Visualización Engañosa: La Taxonomía de Correll, Espejismos de Agregación y Ética Visual (Diaps. 76–103):**
     * **Las 4 categorías de la taxonomía de Michael Correll (Tableau Research, Diap. 77):**
       1. *Incorrectas (codifican mal los datos, Diaps. 78–85):* Hong Kong en Brasil (CNN, 2013), $2385^\circ\text{F}$ en Wickenburg (Fox 10 Phoenix, 2015), barras sin escala en *The Daily Nexus* UCSB ($0.7\%$ y $29.4\%$ con casi igual altura, 2010), el circular que suma $70\% + 63\% + 60\% = \mathbf{193\%}$ (Fox News, 2009) y el doble eje manipulado (ventas nosotros $300 \to 340$ vs. rival $88 \to 101$, y las «tijeras» del *Ithaca Times*, 2000, con 33 años de matrícula vs. 11 años de ranking invertido).
       2. *Ilegibles (ahogan al lector en detalle, Diaps. 86–87):* Circular con 24 porciones y 5 tonos repetidos, y serie temporal de cesantía en Alabama (2021) codificada como burbujas desordenadas.
       3. *Bullshit (indiferencia por la verdad, Harry Frankfurt, Diaps. 88–90):* El mentiroso conoce la verdad y la contradice; el *bullshitter* la ignora. Tres especies: inventar cosas que parecen datos (*Decline of Artistic Standards*), asumir la conclusión y graficarla, o entregar datos inertes sin contexto.
       4. *No convencionales (rompen la convención de lectura, Diaps. 91–95):* Muertes por armas en Florida (Reuters, 2014, con el $0$ arriba donde el pico de $873$ tras la ley *Stand Your Ground* de 2005 parece una caída, frente a la retórica declarada de *Iraq's Bloody Toll* de S. Scarr, 2011) y el mapa de ITS donde el morado oscuro representa *menos* infecciones (violando «más tinta = más»).
     * **Visualizaciones frágiles y espejismos (*Visualization Mirages*, McNutt, Kindlmann y Correll, CHI 2020, Diaps. 96–98):** La suma mundial de energía parece desplomarse en 2012 porque los países sin dato saltan de $\approx 70$ a $\approx 175$; si se cambia `SUM` por `AVG`, 2012 parece dispararse porque solo quedan los países de alto consumo que sí reportaron.
     * **Visualizaciones inhumanas (*Cruel Pies*, Dragga y Voss, 2001: 771 muertes madereras), mentiras blancas y los 4 casos aplicados de la Diapositiva 100.**
* Actualizar `web/index.html` situando **CD03: Visualización de Datos Efectiva y Engañosa (103 Diaps.)** en **tercera posición** dentro de `.unit-switcher` (después de `CD01` y `CD02`, y antes de `CD13`), incorporando 4 nuevos laboratorios interactivos (`panel-33` a `panel-36`) y 4 nuevas preguntas de autoevaluación (`quiz-item-33` a `quiz-item-36`), alcanzando **9 Unidades Académicas (563 Diapositivas) y 37 Laboratorios Interactivos** ordenados por código (`CD01`, `CD02`, `CD03`, `CD13`–`CD18`).
* Integrar `visualizacion_datos.py` en tercera posición dentro de `perceptron_xor.py` (`--viz` / `-v` / `--cd03`, menú interactivo y tercer bloque de `--auto`), `README.md` e `INFORME.md`.

## 2. Plan de Trabajo (Resumen)
* **Objetivo:** Integrar teórica, computacional y visualmente la tercera unidad del curso **CD03 — Visualización de Datos Efectiva y Engañosa (103 Diapositivas)** inmediatamente después de `CD02` y antes de `CD13`.
* **Fases/Hitos principales:**
  1. Creación del registro estructurado `contexto/context14.md`.
  2. Implementación del módulo `visualizacion_datos.py` con verificaciones numéricas exactas de las Diapositivas 8–14, 21, 24–38, 40–49, 56–74 y 76–101.
  3. Actualización de `web/index.html` ubicando **CD03 — Visualización de Datos** como tercera unidad en `.unit-switcher` y añadiendo sus 4 laboratorios (`panel-33` a `panel-36`) y 4 reactivos de autoevaluación (`quiz-item-33` a `quiz-item-36`).
  4. Integración en tercera posición dentro de `perceptron_xor.py`, actualización de `README.md` e `INFORME.md`, y verificación automatizada (`python3 perceptron_xor.py --auto`).
* **Entregables:** `contexto/context14.md`, `visualizacion_datos.py`, `web/index.html` actualizado (9 unidades ordenadas `CD01`, `CD02`, `CD03`, `CD13`–`CD18`; 37 laboratorios), `perceptron_xor.py`, `README.md` e `INFORME.md`.

## 3. Referencias
* **Path del plan original:** `contexto/context14.md`
* **Material de cátedra:** Presentación *«CD03 — Visualización de Datos Efectiva y Engañosa»* (Ciencia de Datos — UCN, 103 diapositivas, basado en CMU 15-388/688 de Z. Kolter y Michael Correll de Tableau Research).
* **Artículos y fuentes fundamentales citados en la unidad CD03:**
  * Matejka, J., & Fitzmaurice, G. (2017). *Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing*. CHI 2017.
  * Burtin, W. (1951). *The effectiveness of penicillin, streptomycin and neomycin on sixteen bacteria*. Scope, Fall 1951; Wainer, H., & Lysen, S. (2009), *American Scientist*, 97(4).
  * Tufte, E. R. (1983). *The Visual Display of Quantitative Information*. Graphics Press.
  * Stevens, S. S. (1961). *To Honor Fechner and Repeal His Law*. Science, 133(3446), 80–86.
  * Cleveland, W. S., & McGill, R. (1984). *Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods*. JASA, 79(387), 531–554.
  * Mackinlay, J. (1986). *Automating the Design of Graphical Presentations of Relational Information*. ACM Transactions on Graphics, 5(2), 110–141.
  * McNutt, A., Kindlmann, G., & Correll, M. (2020). *Surfacing Visualization Mirages*. CHI 2020.
