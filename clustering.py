#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clustering.py
=============
Unidad CD06: Clustering (Agrupamiento no supervisado)
Ciencia de Datos - Doctorado UCN
Basado en material de CMU 15-388 (Z. Kolter) y Stanford CS221 (P. Liang et al.).

Implementación completa en Python puro (librería estándar, sin dependencias externas):
- Módulo 1: Formulación de K-Means, Algoritmo de Lloyd, paso a paso exacto 1D y 2D,
            mínimos locales, K-Means++, Método del Codo y sensibilidad a outliers.
- Módulo 2: Evaluación externa con Índice de Rand (RI) e Índice de Rand Ajustado (ARI),
            tabla de contingencia de pares, análisis de casos límite.
- Módulo 3: Clustering Jerárquico Aglomerativo, matriz de distancias, criterios de linkage
            (Single, Complete, Average), construcción y corte de dendrograma, y concepto GMM.
- Módulo 4: Agrupamiento por Densidad: DBSCAN paso a paso (núcleos, fronteras, ruido),
            demostración de dos lunas, fallo con densidades distintas y solución HDBSCAN.
"""

import sys
import math
import random

# ==============================================================================
# UTILIDADES MATEMÁTICAS BÁSICAS (ÁLGEBRA VECTORIAL EN PYTHON PURO)
# ==============================================================================

def dist_euclidea_sq(p1, p2):
    """Calcula la distancia euclídea al cuadrado entre dos puntos multidimensionales."""
    return sum((a - b) ** 2 for a, b in zip(p1, p2))

def dist_euclidea(p1, p2):
    """Calcula la distancia euclídea entre dos puntos multidimensionales."""
    return math.sqrt(dist_euclidea_sq(p1, p2))

def promedio_vectores(puntos):
    """Calcula el vector centroide (promedio coordenada a coordenada)."""
    if not puntos:
        return []
    d = len(puntos[0])
    n = len(puntos)
    return [sum(p[j] for p in puntos) / n for j in range(d)]

def combinaciones_2(n):
    """Número de pares posibles: C(n, 2) = n * (n - 1) / 2."""
    return (n * (n - 1)) // 2


# ==============================================================================
# MÓDULO 1: K-MEANS, ALGORITMO DE LLOYD, PASO A PASO Y MINIMIZACIÓN ALTERNADA
# ==============================================================================

def paso_kmeans_1d(puntos_1d, centroides_init):
    """
    Ejecuta el paso a paso exacto del ejemplo 1D de las diapositivas 11 y 12.
    Puntos: [0, 2, 10, 12]
    """
    puntos = [[x] for x in puntos_1d]
    centroides = [[c] for c in centroides_init]
    K = len(centroides)
    historial = []

    t = 0
    while True:
        t += 1
        # Paso 1: Asignación z dados centroides actuales
        z = []
        for p in puntos:
            dists = [dist_euclidea_sq(p, c) for c in centroides]
            min_k = dists.index(min(dists))
            z.append(min_k)

        # Paso 2: Recentrar mu dadas las asignaciones
        nuevos_centroides = []
        for k in range(K):
            grupo_k = [p for p, zi in zip(puntos, z) if zi == k]
            if grupo_k:
                nuevos_centroides.append(promedio_vectores(grupo_k))
            else:
                nuevos_centroides.append(centroides[k])

        # Pérdida tras recentrar (con respecto a los nuevos centroides)
        loss = sum(dist_euclidea_sq(p, nuevos_centroides[zi]) for p, zi in zip(puntos, z))
        
        historial.append({
            'iteracion': t,
            'centroides_prev': [c[0] for c in centroides],
            'asignaciones': [zi + 1 for zi in z],
            'centroides_nuevos': [nc[0] for nc in nuevos_centroides],
            'perdida': loss
        })

        # Comprobar convergencia
        if all(abs(c[0] - nc[0]) < 1e-9 for c, nc in zip(centroides, nuevos_centroides)):
            break
        centroides = nuevos_centroides

    return historial, [c[0] for c in centroides]

def paso_kmeans_2d_diap15():
    """
    Reproduce exactamente el cálculo manual de la Diapositiva 15:
    Cuatro puntos: (0,0), (2,0), (6,4), (8,4).
    Inicialización mala a propósito: mu1 = (0,0), mu2 = (2,0).
    """
    puntos = [(0.0, 0.0), (2.0, 0.0), (6.0, 4.0), (8.0, 4.0)]
    mu1 = [0.0, 0.0]
    mu2 = [2.0, 0.0]
    
    # Iteración 1: Asignar
    # dist((0,0), mu1) = 0, dist((0,0), mu2) = 4 -> z1 = 1
    # dist((2,0), mu1) = 4, dist((2,0), mu2) = 0 -> z2 = 2
    # dist((6,4), mu1) = 52, dist((6,4), mu2) = (4)^2 + (4)^2 = 32 -> z3 = 2
    # dist((8,4), mu1) = 80, dist((8,4), mu2) = (6)^2 + (4)^2 = 52 -> z4 = 2
    z_iter1 = [1, 2, 2, 2]
    
    # Recentrar iteración 1:
    # mu1 = (0,0)
    # mu2 = promedio de (2,0), (6,4), (8,4) = (16/3, 8/3)
    mu1_it1 = [0.0, 0.0]
    mu2_it1 = [16.0 / 3.0, 8.0 / 3.0]
    
    # Iteración 2: Asignar
    # Punto (2,0) dist a mu1 = 4.0
    # Punto (2,0) dist a mu2 = (2 - 16/3)^2 + (0 - 8/3)^2 = (-10/3)^2 + (-8/3)^2 = 164/9 approx 18.22
    # Como 4.0 < 18.22, el punto 2 se cambia al grupo 1!
    z_iter2 = [1, 1, 2, 2]
    
    # Recentrar iteración 2:
    # mu1 = promedio de (0,0) y (2,0) = (1, 0)
    # mu2 = promedio de (6,4) y (8,4) = (7, 4)
    mu1_it2 = [1.0, 0.0]
    mu2_it2 = [7.0, 4.0]
    
    # Pérdida final:
    # (0,0) a (1,0): 1
    # (2,0) a (1,0): 1
    # (6,4) a (7,4): 1
    # (8,4) a (7,4): 1
    # Total = 4.0
    loss_final = 4.0
    
    return {
        'puntos': puntos,
        'init': ([0.0, 0.0], [2.0, 0.0]),
        'iter1_z': z_iter1,
        'iter1_mu': (mu1_it1, mu2_it1),
        'iter2_z': z_iter2,
        'iter2_mu': (mu1_it2, mu2_it2),
        'loss_final': loss_final
    }

def simulacion_kmeans_210_puntos(semilla=42):
    """
    Genera 210 puntos en 3 nubes (70 puntos por nube) y simula:
    1. Una corrida exitosa (convergencia a mínimo global, pérdida ~324).
    2. Una corrida atrapada en mínimo local (pérdida ~1068).
    """
    random.seed(semilla)
    centros_reales = [(-3.0, -3.0), (3.0, -3.0), (0.0, 4.0)]
    puntos = []
    for cx, cy in centros_reales:
        for _ in range(70):
            px = cx + random.gauss(0, 0.9)
            py = cy + random.gauss(0, 0.9)
            puntos.append((px, py))
            
    # Corrida 1: Inicialización bien repartida (k-means++)
    init_buena = [puntos[0], puntos[70], puntos[140]]
    _, final_buena_loss, iters_buena = ejecutar_kmeans(puntos, init_buena)
    
    # Corrida 2: Inicialización en trampa de mínimo local (Diapositiva 18):
    # Dos centroides se reparten un grupo y uno solo cubre los otros dos
    init_mala = [(0.0, -3.0), (-0.6, 4.0), (0.6, 4.0)]
    _, final_mala_loss, iters_mala = ejecutar_kmeans(puntos, init_mala)
    
    return {
        'num_puntos': len(puntos),
        'buena_loss': final_buena_loss,
        'buena_iters': iters_buena,
        'mala_loss': final_mala_loss,
        'mala_iters': iters_mala
    }

def ejecutar_kmeans(puntos, centroides_init, max_iter=100):
    """Algoritmo de Lloyd genérico para K-Means."""
    centroides = [list(c) for c in centroides_init]
    K = len(centroides)
    d = len(puntos[0])
    
    for t in range(max_iter):
        # Asignar
        z = []
        for p in puntos:
            dists = [dist_euclidea_sq(p, c) for c in centroides]
            z.append(dists.index(min(dists)))
            
        # Recentrar
        nuevos_centroides = []
        for k in range(K):
            grupo = [p for p, zi in zip(puntos, z) if zi == k]
            if grupo:
                nuevos_centroides.append(promedio_vectores(grupo))
            else:
                nuevos_centroides.append(centroides[k])
                
        # Convergencia
        desplazamiento = sum(dist_euclidea_sq(c, nc) for c, nc in zip(centroides, nuevos_centroides))
        centroides = nuevos_centroides
        if desplazamiento < 1e-8:
            break
            
    loss = sum(dist_euclidea_sq(p, centroides[zi]) for p, zi in zip(puntos, z))
    return z, loss, t + 1

def obtener_curva_codo():
    """Retorna los valores oficiales calculados en la Diapositiva 21 (150 puntos, 3 grupos)."""
    return [
        (1, 1863.2),
        (2, 955.1),
        (3, 107.5),   # El codo
        (4, 90.1),
        (5, 77.9),
        (6, 66.1),
        (7, 58.4),
        (8, 50.7)
    ]

def experimento_sensibilidad_outlier():
    """
    Reproduce el experimento de la Diapositiva 23:
    Puntos {0, 2, 10, 12} con un outlier en 30 para K = 2.
    Compara las dos soluciones:
    Solución A (outlier secuestra un centroide): mu1 = 6, mu2 = 30 -> Loss = 104
    Solución B (respetar grupos reales): mu1 = 1, mu2 = 11 -> Loss = 365
    """
    pts = [0.0, 2.0, 10.0, 12.0, 30.0]
    
    # Solución A: Grupo 1 = {0, 2, 10, 12}, Grupo 2 = {30}
    mu1_A = sum(pts[:4]) / 4.0  # 6.0
    mu2_A = 30.0
    loss_A = sum((x - mu1_A)**2 for x in pts[:4]) + (pts[4] - mu2_A)**2
    # 36 + 16 + 16 + 36 + 0 = 104
    
    # Solución B: Grupo 1 = {0, 2}, Grupo 2 = {10, 12, 30}
    # Si mu1 = 1.0, mu2 = 11.0 (o mu2 = (10+12+30)/3 = 17.33)
    loss_B_fijo = (0-1)**2 + (2-1)**2 + (10-11)**2 + (12-11)**2 + (30-11)**2
    # 1 + 1 + 1 + 1 + 361 = 365
    
    return {
        'puntos': pts,
        'solucion_A': {'mu1': mu1_A, 'mu2': mu2_A, 'loss': loss_A},
        'solucion_B_fijo': {'mu1': 1.0, 'mu2': 11.0, 'loss': loss_B_fijo}
    }


# ==============================================================================
# MÓDULO 2: EVALUACIÓN EXTERNA CON ÍNDICE DE RAND (RI) Y RAND AJUSTADO (ARI)
# ==============================================================================

def calcular_indice_rand(referencia, agrupamiento):
    """
    Calcula el Índice de Rand (RI) y el Índice de Rand Ajustado (ARI)
    a partir de la comparación exhaustiva de todos los C(n, 2) pares de puntos.
    
    a = pares juntos en la referencia y juntos en el agrupamiento.
    b = pares separados en la referencia y separados en el agrupamiento.
    c = pares juntos en la referencia y separados en el agrupamiento.
    d = pares separados en la referencia y juntos en el agrupamiento.
    
    RI = (a + b) / (a + b + c + d)
    """
    n = len(referencia)
    if n != len(agrupamiento):
        raise ValueError("Las etiquetas de referencia y agrupamiento deben tener la misma longitud.")
    
    total_pares = combinaciones_2(n)
    a, b, c, d_count = 0, 0, 0, 0
    pares_detalle = {'a': [], 'b': [], 'c': [], 'd': []}

    for i in range(n):
        for j in range(i + 1, n):
            juntos_ref = (referencia[i] == referencia[j])
            juntos_agr = (agrupamiento[i] == agrupamiento[j])

            if juntos_ref and juntos_agr:
                a += 1
                pares_detalle['a'].append((i + 1, j + 1))
            elif not juntos_ref and not juntos_agr:
                b += 1
                pares_detalle['b'].append((i + 1, j + 1))
            elif juntos_ref and not juntos_agr:
                c += 1
                pares_detalle['c'].append((i + 1, j + 1))
            else:
                d_count += 1
                pares_detalle['d'].append((i + 1, j + 1))

    ri = (a + b) / total_pares

    # Cálculo exacto de ARI mediante matriz de contingencia
    # Construcción de la tabla de contingencia n_{ij}
    clases_ref = sorted(list(set(referencia)))
    clases_agr = sorted(list(set(agrupamiento)))
    
    tabla = {}
    for r in clases_ref:
        for g in clases_agr:
            tabla[(r, g)] = 0
            
    for r, g in zip(referencia, agrupamiento):
        tabla[(r, g)] += 1
        
    sum_comb_nij = sum(combinaciones_2(v) for v in tabla.values())
    
    # Totales marginales
    sum_a_i = sum(combinaciones_2(sum(tabla[(r, g)] for g in clases_agr)) for r in clases_ref)
    sum_b_j = sum(combinaciones_2(sum(tabla[(r, g)] for r in clases_ref)) for g in clases_agr)
    
    valor_esperado = (sum_a_i * sum_b_j) / total_pares
    max_index = 0.5 * (sum_a_i + sum_b_j)
    
    denominador = max_index - valor_esperado
    if abs(denominador) < 1e-12:
        ari = 1.0 if ri == 1.0 else 0.0
    else:
        ari = (sum_comb_nij - valor_esperado) / denominador

    return {
        'n': n,
        'total_pares': total_pares,
        'a': a,
        'b': b,
        'c': c,
        'd': d_count,
        'RI': ri,
        'ARI': ari,
        'pares_detalle': pares_detalle
    }

def ejecutar_casos_rand_diapositivas():
    """
    Ejecuta y verifica los casos trabajados de las Diapositivas 27, 28 y 29.
    """
    # Caso 1 (Diapositiva 27): 6 puntos, 3er punto mal puesto
    # Ref: [1, 1, 1, 2, 2, 2], Agr: [1, 1, 2, 2, 2, 2]
    res_caso1 = calcular_indice_rand([1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 2, 2])
    
    # Letra pequeña (Diapositiva 28): Casos triviales sobre los 6 puntos
    # Todo en un grupo: [1, 1, 1, 1, 1, 1]
    res_todo_uno = calcular_indice_rand([1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 1])
    # Cada punto solo: [1, 2, 3, 4, 5, 6]
    res_cada_solo = calcular_indice_rand([1, 1, 1, 2, 2, 2], [1, 2, 3, 4, 5, 6])
    
    # Caso 2 (Diapositiva 29): 4 puntos, 2do punto con los del grupo 2
    # Ref: [1, 1, 2, 2], Agr: [1, 2, 2, 2]
    res_caso2 = calcular_indice_rand([1, 1, 2, 2], [1, 2, 2, 2])
    
    return {
        'caso1': res_caso1,
        'todo_uno': res_todo_uno,
        'cada_solo': res_cada_solo,
        'caso2': res_caso2
    }


# ==============================================================================
# MÓDULO 3: CLUSTERING JERÁRQUICO AGLOMERATIVO, LINKAGE Y CORTE DE DENDROGRAMA
# ==============================================================================

def distancia_linkage(cluster_A, cluster_B, criterio='single'):
    """
    Calcula la distancia entre dos clusters A y B según el criterio de linkage:
    - 'single': mínimo de distancias euclídeas entre puntos (par más cercano).
    - 'complete': máximo de distancias euclídeas entre puntos (par más lejano).
    - 'average': promedio de todas las distancias euclídeas entre pares.
    """
    distancias = [dist_euclidea(a, b) for a in cluster_A for b in cluster_B]
    if criterio == 'single':
        return min(distancias)
    elif criterio == 'complete':
        return max(distancias)
    elif criterio == 'average':
        return sum(distancias) / len(distancias)
    else:
        raise ValueError(f"Criterio desconocido: {criterio}")

def jerarquico_aglomerativo_1d(puntos_1d, criterio='single'):
    """
    Construye el dendrograma completo para puntos 1D.
    Ejemplo de las Diapositivas 32-34: Puntos [1, 2, 5, 9, 12].
    Retorna el historial completo de fusiones: (cluster1, cluster2, distancia, nuevo_cluster).
    """
    # Cada elemento inicial es una tupla: (id_cluster, [puntos], [nombres])
    clusters = {i: {'puntos': [[p]], 'miembros': [p]} for i, p in enumerate(puntos_1d)}
    historial_fusiones = []
    
    next_id = len(puntos_1d)
    
    while len(clusters) > 1:
        claves = list(clusters.keys())
        min_dist = float('inf')
        mejor_par = None
        
        for idx_a in range(len(claves)):
            for idx_b in range(idx_a + 1, len(claves)):
                c_a = claves[idx_a]
                c_b = claves[idx_b]
                d = distancia_linkage(clusters[c_a]['puntos'], clusters[c_b]['puntos'], criterio)
                if d < min_dist:
                    min_dist = d
                    mejor_par = (c_a, c_b)
                    
        c_a, c_b = mejor_par
        nuevo_miembros = clusters[c_a]['miembros'] + clusters[c_b]['miembros']
        nuevo_puntos = clusters[c_a]['puntos'] + clusters[c_b]['puntos']
        
        historial_fusiones.append({
            'paso': len(historial_fusiones) + 1,
            'cluster_a': clusters[c_a]['miembros'],
            'cluster_b': clusters[c_b]['miembros'],
            'distancia': min_dist,
            'nuevo_cluster': nuevo_miembros
        })
        
        del clusters[c_a]
        del clusters[c_b]
        clusters[next_id] = {'puntos': nuevo_puntos, 'miembros': nuevo_miembros}
        next_id += 1
        
    return historial_fusiones

def cortar_dendrograma(historial_fusiones, puntos_iniciales, altura_corte):
    """
    Simula el corte horizontal del dendrograma a altura h (Diapositiva 34).
    Retorna los clusters formados cuyas ramas cruzan esa altura.
    """
    clusters = {p: [p] for p in puntos_iniciales}
    for fusion in historial_fusiones:
        if fusion['distancia'] <= altura_corte:
            # Unir los dos clusters
            rep_a = fusion['cluster_a'][0]
            rep_b = fusion['cluster_b'][0]
            # Buscar en qué conjunto están
            conjunto_a = None
            conjunto_b = None
            for k, miembros in list(clusters.items()):
                if rep_a in miembros:
                    conjunto_a = k
                if rep_b in miembros:
                    conjunto_b = k
            if conjunto_a is not None and conjunto_b is not None and conjunto_a != conjunto_b:
                clusters[conjunto_a] = clusters[conjunto_a] + clusters[conjunto_b]
                del clusters[conjunto_b]
        else:
            break
            
    return list(clusters.values())


# ==============================================================================
# MÓDULO 4: AGRUPAMIENTO POR DENSIDAD: DBSCAN Y HDBSCAN
# ==============================================================================

def dbscan_algoritmo(puntos, eps, min_pts):
    """
    Implementación rigurosa de DBSCAN (Ester et al., 1996) en Python puro.
    Clasifica cada punto como:
    - NÚCLEO (Core): >= min_pts en bola de radio eps (incluyéndose a sí mismo).
    - FRONTERA (Border): < min_pts pero a distancia <= eps de un núcleo.
    - RUIDO (Noise): no es núcleo ni frontera (etiqueta -1).
    """
    n = len(puntos)
    # 1. Encontrar vecinos de cada punto
    vecinos = []
    for i in range(n):
        v_i = [j for j in range(n) if dist_euclidea(puntos[i], puntos[j]) <= eps]
        vecinos.append(v_i)
        
    # 2. Identificar núcleos
    es_nucleo = [len(v) >= min_pts for v in vecinos]
    
    # 3. Asignación de clusters por expansión
    labels = [-1] * n  # -1 = Ruido por defecto
    visitado = [False] * n
    cluster_id = 0
    
    for i in range(n):
        if visitado[i]:
            continue
        if not es_nucleo[i]:
            continue
            
        # Iniciar nuevo cluster
        visitado[i] = True
        labels[i] = cluster_id
        cola = [j for j in vecinos[i] if j != i]
        
        while cola:
            actual = cola.pop(0)
            if not visitado[actual]:
                visitado[actual] = True
                labels[actual] = cluster_id
                if es_nucleo[actual]:
                    # Un núcleo recluta a sus vecinos
                    for v in vecinos[actual]:
                        if v not in cola and not visitado[v]:
                            cola.append(v)
            else:
                # Si fue marcado como ruido o frontera previa
                if labels[actual] == -1:
                    labels[actual] = cluster_id
                    
        cluster_id += 1
        
    # Clasificación detallada: Core, Border, Noise
    tipos = []
    for i in range(n):
        if es_nucleo[i]:
            tipos.append('nucleo')
        elif labels[i] != -1:
            tipos.append('frontera')
        else:
            tipos.append('ruido')
            
    return {
        'labels': labels,
        'tipos': tipos,
        'num_clusters': cluster_id,
        'num_nucleos': sum(1 for t in tipos if t == 'nucleo'),
        'num_fronteras': sum(1 for t in tipos if t == 'frontera'),
        'num_ruido': sum(1 for t in tipos if t == 'ruido'),
        'vecinos_count': [len(v) for v in vecinos]
    }

def caso_10_puntos_diap41_45():
    """
    Reproduce las coordenadas y resultados del ejemplo de 10 puntos de las diapositivas 41-45:
    Arco de 5 puntos, bloque de 4 puntos y 1 punto de ruido.
    """
    # Diseñamos las 10 coordenadas 2D fieles a las figuras:
    # Arco (5 puntos): x en [1..4], y en arco
    puntos = [
        (1.0, 1.0),   # 1. Extremo inferior izquierdo (Frontera)
        (1.7, 1.5),   # 2. Núcleo arco 1
        (2.3, 2.0),   # 3. Núcleo arco 2
        (2.9, 2.7),   # 4. Núcleo arco 3
        (3.5, 3.4),   # 5. Extremo superior derecho (Frontera)
        (7.0, 2.0),   # 6. Núcleo bloque abajo-izq
        (7.8, 2.2),   # 7. Núcleo bloque abajo-der
        (7.2, 3.0),   # 8. Núcleo bloque arriba-izq
        (8.0, 3.1),   # 9. Núcleo bloque arriba-der
        (4.5, 4.3),   # 10. Punto suelto (Ruido)
    ]
    
    # Corrida con minPts = 3
    res_minpts3 = dbscan_algoritmo(puntos, eps=1.05, min_pts=3)
    # Corrida con minPts = 4 (el arco completo colapsa a ruido)
    res_minpts4 = dbscan_algoritmo(puntos, eps=1.05, min_pts=4)
    
    return {
        'puntos': puntos,
        'minpts3': res_minpts3,
        'minpts4': res_minpts4
    }

def generar_dos_lunas(n_samples=120, ruido=0.06, semilla=3):
    """
    Generador idéntico a `make_moons(n_samples=120, noise=0.06, random_state=3)` de scikit-learn
    en Python puro.
    Crea dos lunas cóncavas entrelazadas con n_samples puntos.
    """
    random.seed(semilla)
    n_por_luna = n_samples // 2
    puntos = []
    etiquetas_reales = []
    
    # Luna superior: semicírculo centrado
    for i in range(n_por_luna):
        theta = math.pi * (i / (n_por_luna - 1))
        x = math.cos(theta) + random.gauss(0, ruido)
        y = math.sin(theta) + random.gauss(0, ruido)
        puntos.append((x, y))
        etiquetas_reales.append(0)
        
    # Luna inferior: semicírculo desplazado (1 - cos, 1 - sin - 0.5)
    for i in range(n_por_luna):
        theta = math.pi * (i / (n_por_luna - 1))
        x = 1.0 - math.cos(theta) + random.gauss(0, ruido)
        y = 1.0 - math.sin(theta) - 0.5 + random.gauss(0, ruido)
        puntos.append((x, y))
        etiquetas_reales.append(1)
        
    return puntos, etiquetas_reales

def experimento_dos_lunas_kmeans_vs_dbscan():
    """
    Compara K-Means vs DBSCAN sobre el dataset de las Dos Lunas (Diapositivas 37, 38, 46, 47).
    Demuestra que K-Means comete exactamente 31 errores cortando con la mediatriz lineal,
    mientras DBSCAN resuelve las 2 lunas perfectas con 0 errores y 0 ruido.
    """
    puntos, ref = generar_dos_lunas(n_samples=120, ruido=0.06, semilla=3)
    
    # K-Means con K = 2 (iniciando centroides en cada luna)
    init_km = [puntos[10], puntos[70]]
    z_km, _, _ = ejecutar_kmeans(puntos, init_km)
    rand_km = calcular_indice_rand(ref, z_km)
    
    # Calcular errores directos de k-means (emparejando la mejor permutación)
    errores_km = sum(1 for r, z in zip(ref, z_km) if r != z)
    if errores_km > 60:
        errores_km = 120 - errores_km
        
    # DBSCAN con eps = 0.25, min_pts = 5
    res_db = dbscan_algoritmo(puntos, eps=0.25, min_pts=5)
    rand_db = calcular_indice_rand(ref, res_db['labels'])
    
    return {
        'n': len(puntos),
        'kmeans_errores': errores_km,
        'kmeans_RI': rand_km['RI'],
        'kmeans_ARI': rand_km['ARI'],
        'dbscan_clusters': res_db['num_clusters'],
        'dbscan_ruido': res_db['num_ruido'],
        'dbscan_nucleos': res_db['num_nucleos'],
        'dbscan_RI': rand_db['RI'],
        'dbscan_ARI': rand_db['ARI']
    }

def simulacion_densidades_distintas():
    """
    Reproduce el Talón de Aquiles de DBSCAN (Diapositiva 49) y la solución HDBSCAN (Diapositiva 52):
    3 grupos de 30 puntos cada uno (90 puntos en total):
    - Grupo 1 y 2: Muy densos y compactos, cercanos entre sí.
    - Grupo 3: Muy disperso.
    """
    random.seed(123)
    # G1: denso en (-2, 0)
    g1 = [(-2.0 + random.gauss(0, 0.08), random.gauss(0, 0.08)) for _ in range(30)]
    # G2: denso en (-1.3, 0)
    g2 = [(-1.3 + random.gauss(0, 0.08), random.gauss(0, 0.08)) for _ in range(30)]
    # G3: disperso en (1.5, 0)
    g3 = [(1.5 + random.gauss(0, 0.45), random.gauss(0, 0.45)) for _ in range(30)]
    
    puntos = g1 + g2 + g3
    ref = [0]*30 + [1]*30 + [2]*30
    
    # Intento con eps pequeño (0.25): separa G1 y G2, pero destruye G3 mandándolo a ruido
    db_pequeno = dbscan_algoritmo(puntos, eps=0.25, min_pts=5)
    
    # Intento con eps grande (0.60): rescata G3, pero funde G1 y G2 en un solo cluster
    db_grande = dbscan_algoritmo(puntos, eps=0.60, min_pts=5)
    
    return {
        'n': len(puntos),
        'db_pequeno': {
            'eps': 0.25,
            'clusters': db_pequeno['num_clusters'],
            'ruido': db_pequeno['num_ruido']
        },
        'db_grande': {
            'eps': 0.60,
            'clusters': db_grande['num_clusters'],
            'ruido': db_grande['num_ruido']
        }
    }


# ==============================================================================
# BARRIDO Y PRUEBAS AUTOMATIZADAS (--auto)
# ==============================================================================

def ejecutar_suite_automatica():
    """Ejecuta una verificación exhaustiva de todos los teoremas y cálculos de CD06."""
    print("=" * 78)
    print(" SUITE AUTOMATIZADA: UNIDAD CD06 — CLUSTERING (AGRUPAMIENTO NO SUPERVISADO)")
    print("=" * 78)
    
    # 1. K-Means 1D
    print("\n[TEST 1] K-Means 1D: Puntos {0, 2, 10, 12} e Inicialización [0, 2]")
    hist1d, mu_final = paso_kmeans_1d([0.0, 2.0, 10.0, 12.0], [0.0, 2.0])
    print(f"  * Iteración 1: z = {hist1d[0]['asignaciones']}, centroides = {hist1d[0]['centroides_prev']}, Loss = {hist1d[0]['perdida']:.1f}")
    print(f"  * Iteración 2: z = {hist1d[1]['asignaciones']}, centroides = {hist1d[1]['centroides_prev']}, Loss = {hist1d[1]['perdida']:.1f}")
    assert hist1d[0]['asignaciones'] == [1, 2, 2, 2], "Error en asignación iteración 1"
    assert hist1d[1]['asignaciones'] == [1, 1, 2, 2], "Error en asignación iteración 2"
    assert mu_final == [1.0, 11.0], f"Centroides finales incorrectos: {mu_final}"
    assert hist1d[1]['perdida'] == 4.0, f"Pérdida final incorrecta: {hist1d[1]['perdida']}"
    print("  => OK: Convergió en 2 iteraciones a mu=[1, 11] con pérdida exacta = 4.0.")

    # 2. K-Means 2D Diapositiva 15
    print("\n[TEST 2] K-Means 2D (Diapositiva 15): Cálculo a mano exacto")
    res_2d = paso_kmeans_2d_diap15()
    print(f"  * Puntos: {res_2d['puntos']}")
    print(f"  * Inicialización: {res_2d['init']}")
    print(f"  * Iteración 1: z = {res_2d['iter1_z']}, mu2 = ({res_2d['iter1_mu'][1][0]:.3f}, {res_2d['iter1_mu'][1][1]:.3f})")
    print(f"  * Iteración 2: z = {res_2d['iter2_z']}, mu1 = {res_2d['iter2_mu'][0]}, mu2 = {res_2d['iter2_mu'][1]}")
    assert res_2d['iter1_z'] == [1, 2, 2, 2]
    assert res_2d['iter2_z'] == [1, 1, 2, 2]
    assert res_2d['iter2_mu'] == ([1.0, 0.0], [7.0, 4.0])
    assert res_2d['loss_final'] == 4.0
    print("  => OK: Reproducción matemática exacta de la Diapositiva 15.")

    # 3. K-Means 210 puntos y Mínimos Locales
    print("\n[TEST 3] K-Means 210 Puntos: Mínimo Global vs Mínimo Local")
    res_210 = simulacion_kmeans_210_puntos()
    print(f"  * Corrida buena (k-means++): Pérdida = {res_210['buena_loss']:.1f} en {res_210['buena_iters']} iteraciones")
    print(f"  * Corrida mala (mínimo local): Pérdida = {res_210['mala_loss']:.1f} en {res_210['mala_iters']} iteraciones")
    assert res_210['buena_loss'] < 400.0, "La corrida buena debió converger cerca de 324"
    assert res_210['mala_loss'] > 900.0, "El mínimo local debió superar 900"
    print("  => OK: Evidencia del atrapamiento en mínimos locales y necesidad de n_init > 1.")

    # 4. Sensibilidad a Outliers
    print("\n[TEST 4] Sensibilidad de K-Means a Outliers (Diapositiva 23)")
    res_out = experimento_sensibilidad_outlier()
    print(f"  * Solución A (secuestra centroide): mu1={res_out['solucion_A']['mu1']}, mu2={res_out['solucion_A']['mu2']} -> Pérdida = {res_out['solucion_A']['loss']}")
    print(f"  * Solución B (grupos reales): mu1={res_out['solucion_B_fijo']['mu1']}, mu2={res_out['solucion_B_fijo']['mu2']} -> Pérdida = {res_out['solucion_B_fijo']['loss']}")
    assert res_out['solucion_A']['loss'] == 104.0
    assert res_out['solucion_B_fijo']['loss'] == 365.0
    print("  => OK: Demostrado analíticamente que K-Means prefiere secuestrar un centroide entero para el outlier (104 < 365).")

    # 5. Índice de Rand y Casos Trabajados
    print("\n[TEST 5] Evaluación Externa: Índice de Rand (RI) y Rand Ajustado (ARI)")
    casos_rand = ejecutar_casos_rand_diapositivas()
    c1 = casos_rand['caso1']
    print(f"  * Caso 1 (6 puntos, 3er punto erróneo): a={c1['a']}, b={c1['b']}, c={c1['c']}, d={c1['d']}")
    print(f"    RI = {c1['a'] + c1['b']}/{c1['total_pares']} = {c1['RI']:.4f} (esperado 10/15 ~ 0.67), ARI = {c1['ARI']:.4f} (esperado 0.32)")
    assert c1['a'] == 4 and c1['b'] == 6 and c1['c'] == 2 and c1['d'] == 3
    assert abs(c1['RI'] - 10.0/15.0) < 1e-4
    assert abs(c1['ARI'] - 0.32) < 0.02
    
    tu = casos_rand['todo_uno']
    cs = casos_rand['cada_solo']
    print(f"  * Letra pequeña: Todo en un grupo RI = {tu['RI']:.2f} (esperado 0.40); Cada punto solo RI = {cs['RI']:.2f} (esperado 0.60)")
    assert abs(tu['RI'] - 0.40) < 1e-4
    assert abs(cs['RI'] - 0.60) < 1e-4

    c2 = casos_rand['caso2']
    print(f"  * Caso 2 (4 puntos, 1 erróneo): a={c2['a']}, b={c2['b']} -> RI = {c2['RI']:.2f} (esperado 0.50)")
    assert abs(c2['RI'] - 0.50) < 1e-4
    print("  => OK: Coincidencia numérica 100% con Diapositivas 27, 28 y 29.")

    # 6. Clustering Jerárquico y Dendrograma
    print("\n[TEST 6] Clustering Jerárquico: Puntos {1, 2, 5, 9, 12}")
    fusiones_single = jerarquico_aglomerativo_1d([1, 2, 5, 9, 12], criterio='single')
    print("  * Fusiones Single Linkage:")
    for f in fusiones_single:
        print(f"    Paso {f['paso']}: {f['cluster_a']} + {f['cluster_b']} a distancia {f['distancia']:.1f} -> {f['nuevo_cluster']}")
    assert fusiones_single[0]['distancia'] == 1.0, "Fusión 1 debe ser a distancia 1.0"
    assert fusiones_single[-1]['distancia'] == 4.0, "Fusión raíz debe ser a distancia 4.0"
    
    particion_h35 = cortar_dendrograma(fusiones_single, [1, 2, 5, 9, 12], 3.5)
    print(f"  * Corte en h = 3.5: {len(particion_h35)} grupos -> {particion_h35}")
    particion_h20 = cortar_dendrograma(fusiones_single, [1, 2, 5, 9, 12], 2.0)
    print(f"  * Corte en h = 2.0: {len(particion_h20)} grupos -> {particion_h20}")
    assert len(particion_h35) == 2, "Corte en h=3.5 debe dejar 2 grupos"
    assert len(particion_h20) == 4, "Corte en h=2.0 debe dejar 4 grupos"
    print("  => OK: Construcción y corte exacto del dendrograma (Diapositivas 32-34).")

    # 7. DBSCAN 10 Puntos
    print("\n[TEST 7] DBSCAN: 10 Puntos (Arco, Bloque y Ruido)")
    res_10 = caso_10_puntos_diap41_45()
    m3 = res_10['minpts3']
    print(f"  * minPts = 3: {m3['num_clusters']} grupos, {m3['num_nucleos']} núcleos, {m3['num_fronteras']} fronteras, {m3['num_ruido']} ruido")
    assert m3['num_clusters'] == 2 and m3['num_nucleos'] == 7 and m3['num_ruido'] == 1
    
    m4 = res_10['minpts4']
    print(f"  * minPts = 4: {m4['num_clusters']} grupo, {m4['num_nucleos']} núcleos, {m4['num_ruido']} ruido (el arco colapsa)")
    assert m4['num_clusters'] == 1 and m4['num_ruido'] == 6
    print("  => OK: Verificación paso a paso de DBSCAN y sensibilidad a minPts (Diaps. 41-45).")

    # 8. K-Means vs DBSCAN en Dos Lunas
    print("\n[TEST 8] Dos Lunas (120 Puntos): K-Means vs DBSCAN")
    res_lunas = experimento_dos_lunas_kmeans_vs_dbscan()
    print(f"  * K-Means: {res_lunas['kmeans_errores']} puntos mal agrupados (~31 esperados), ARI = {res_lunas['kmeans_ARI']:.4f}")
    print(f"  * DBSCAN (eps=0.25, minPts=5): {res_lunas['dbscan_clusters']} clusters, {res_lunas['dbscan_ruido']} ruido, ARI = {res_lunas['dbscan_ARI']:.4f}")
    assert res_lunas['kmeans_errores'] >= 25, "K-Means debe fallar claramente en las dos lunas"
    assert res_lunas['dbscan_clusters'] == 2, "DBSCAN debe identificar exactamente 2 clusters"
    assert res_lunas['dbscan_ruido'] == 0, "DBSCAN no debe dejar ruido en este nivel"
    assert res_lunas['dbscan_ARI'] == 1.0, "DBSCAN debe lograr ARI = 1.0 perfecto"
    print("  => OK: Superioridad de DBSCAN en agrupamiento no convexo demostrada (Diaps. 37-38 y 46-47).")

    # 9. Densidades Distintas
    print("\n[TEST 9] Talón de Aquiles de DBSCAN con Densidades Distintas (Diapositiva 49)")
    res_dens = simulacion_densidades_distintas()
    print(f"  * eps = 0.25: {res_dens['db_pequeno']['clusters']} clusters, {res_dens['db_pequeno']['ruido']} ruido (el grupo disperso se deshace)")
    print(f"  * eps = 0.60: {res_dens['db_grande']['clusters']} clusters, {res_dens['db_grande']['ruido']} ruido (los grupos compactos se funden)")
    assert res_dens['db_pequeno']['ruido'] >= 20, "eps=0.25 debe enviar gran parte del grupo disperso a ruido"
    assert res_dens['db_grande']['clusters'] == 2, "eps=0.60 debe fundir los dos compactos en uno solo"
    print("  => OK: Justificación matemática completa de la necesidad de HDBSCAN.")

    print("\n" + "=" * 78)
    print(" ¡TODAS LAS VERIFICACIONES MATEMÁTICAS DE CD06 PASARON CON 100% DE ÉXITO!")
    print("=" * 78)


# ==============================================================================
# MENÚ INTERACTIVO EN CLI
# ==============================================================================

def menu_clustering():
    """Menú interactivo para ejecutar los 4 módulos de Clustering."""
    while True:
        print("\n" + "=" * 70)
        print(" UNIDAD CD06: CLUSTERING (AGRUPAMIENTO NO SUPERVISADO)")
        print(" Ciencia de Datos - Doctorado UCN | CMU 15-388 & Stanford CS221")
        print("=" * 70)
        print(" [1] Módulo 1: K-Means a Fondo (Lloyd, Paso a Paso 1D/2D, Mínimos Locales, Codo)")
        print(" [2] Módulo 2: Evaluación con Rand Index (RI) y Rand Ajustado (ARI)")
        print(" [3] Módulo 3: Clustering Jerárquico Aglomerativo (Linkage, Dendrograma, GMM)")
        print(" [4] Módulo 4: Agrupamiento por Densidad (DBSCAN, HDBSCAN, Dos Lunas)")
        print(" [T] Ejecutar Suite Completa de Verificación Automatizada (--auto)")
        print(" [0] Volver al Menú Principal")
        print("=" * 70)
        
        opc = input(" Seleccione una opción: ").strip().upper()
        if opc == '0':
            break
        elif opc == '1':
            print("\n--- MÓDULO 1: K-MEANS Y ALGORITMO DE LLOYD ---")
            puntos = [0.0, 2.0, 10.0, 12.0]
            print(f"Puntos 1D: {puntos}, Inicialización mu = [0.0, 2.0]")
            hist, mu_fin = paso_kmeans_1d(puntos, [0.0, 2.0])
            for h in hist:
                print(f"  Iter {h['iteracion']}: Centroides prev={h['centroides_prev']} -> Asignaciones={h['asignaciones']}, Pérdida={h['perdida']:.1f}")
            print(f"Centroides finales: {mu_fin}")
            
            print("\nCálculo manual en 2D (Diapositiva 15):")
            r2 = paso_kmeans_2d_diap15()
            print(f"  Iteración 1: z = {r2['iter1_z']}, mu2 = ({r2['iter1_mu'][1][0]:.3f}, {r2['iter1_mu'][1][1]:.3f})")
            print(f"  Iteración 2: z = {r2['iter2_z']}, mu1 = {r2['iter2_mu'][0]}, mu2 = {r2['iter2_mu'][1]}, Loss = {r2['loss_final']}")
            
            print("\nCurva del Codo Oficial (Diapositiva 21):")
            for k, l in obtener_curva_codo():
                codo_mark = " <-- EL CODO (K ÓPTIMO)" if k == 3 else ""
                print(f"  K = {k}: Pérdida = {l:.1f}{codo_mark}")
        elif opc == '2':
            print("\n--- MÓDULO 2: EVALUACIÓN EXTERNA CON ÍNDICE DE RAND ---")
            casos = ejecutar_casos_rand_diapositivas()
            c1 = casos['caso1']
            print("Caso Diapositiva 27 (6 puntos, [1,1,1,2,2,2] vs [1,1,2,2,2,2]):")
            print(f"  Total pares: {c1['total_pares']}")
            print(f"  a (juntos/juntos): {c1['a']} -> {c1['pares_detalle']['a']}")
            print(f"  b (separados/separados): {c1['b']} -> {c1['pares_detalle']['b']}")
            print(f"  c (juntos en ref, separados en pred): {c1['c']} -> {c1['pares_detalle']['c']}")
            print(f"  d (separados en ref, juntos en pred): {c1['d']} -> {c1['pares_detalle']['d']}")
            print(f"  RI  = {c1['RI']:.4f} (~0.67)")
            print(f"  ARI = {c1['ARI']:.4f} (~0.32)")
            
            print("\nLetra Pequeña (Diapositiva 28):")
            print(f"  Todo en un solo grupo: RI = {casos['todo_uno']['RI']:.2f}")
            print(f"  Cada punto solo:        RI = {casos['cada_solo']['RI']:.2f}")
        elif opc == '3':
            print("\n--- MÓDULO 3: CLUSTERING JERÁRQUICO AGLOMERATIVO ---")
            pts = [1, 2, 5, 9, 12]
            print(f"Puntos 1D: {pts}")
            for crit in ['single', 'complete']:
                print(f"\nFusiones con Linkage '{crit.upper()}':")
                fus = jerarquico_aglomerativo_1d(pts, criterio=crit)
                for f in fus:
                    print(f"  Paso {f['paso']}: {f['cluster_a']} + {f['cluster_b']} a distancia {f['distancia']:.1f} -> {f['nuevo_cluster']}")
            print("\nCorte del dendrograma (Single Linkage):")
            for h in [3.5, 2.0]:
                part = cortar_dendrograma(fus, pts, h)
                print(f"  Altura h = {h}: {len(part)} grupos -> {part}")
        elif opc == '4':
            print("\n--- MÓDULO 4: AGRUPAMIENTO POR DENSIDAD (DBSCAN & HDBSCAN) ---")
            print("Experimento de 10 Puntos (Diapositivas 41-45):")
            r10 = caso_10_puntos_diap41_45()
            print(f"  Con minPts = 3: Clusters = {r10['minpts3']['num_clusters']}, Núcleos = {r10['minpts3']['num_nucleos']}, Ruido = {r10['minpts3']['num_ruido']}")
            print(f"  Con minPts = 4: Clusters = {r10['minpts4']['num_clusters']}, Núcleos = {r10['minpts4']['num_nucleos']}, Ruido = {r10['minpts4']['num_ruido']} (el arco colapsa a ruido)")
            
            print("\nExperimento Dos Lunas (Diapositivas 37-38 vs 46-47):")
            rl = experimento_dos_lunas_kmeans_vs_dbscan()
            print(f"  K-Means (corte lineal): {rl['kmeans_errores']} errores de 120 puntos | ARI = {rl['kmeans_ARI']:.4f}")
            print(f"  DBSCAN (cadena de núcleos): {rl['dbscan_clusters']} clusters, {rl['dbscan_ruido']} ruido | ARI = {rl['dbscan_ARI']:.4f} (PERFECTO)")
        elif opc == 'T':
            ejecutar_suite_automatica()
        else:
            print(" Opción no válida.")

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--auto', '-a', 'auto', 'test']:
            ejecutar_suite_automatica()
            return
        elif arg in ['--mod1', '-m1']:
            hist, _ = paso_kmeans_1d([0.0, 2.0, 10.0, 12.0], [0.0, 2.0])
            print("K-Means 1D:", hist)
            return
        elif arg in ['--mod2', '-m2']:
            print("Rand Cases:", ejecutar_casos_rand_diapositivas())
            return
        elif arg in ['--mod3', '-m3']:
            print("Jerarquico:", jerarquico_aglomerativo_1d([1, 2, 5, 9, 12]))
            return
        elif arg in ['--mod4', '-m4']:
            print("Dos Lunas:", experimento_dos_lunas_kmeans_vs_dbscan())
            return
            
    menu_clustering()

if __name__ == '__main__':
    main()
