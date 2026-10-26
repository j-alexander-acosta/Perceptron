#!/usr/bin/env python3
"""
Entorno Gráfico de Escritorio para el Perceptrón Multicapa XOR
Construido con Python y Tkinter (interfaz nativa multiplataforma).

Permite manipular interactivamente las entradas X e Y,
visualizando en tiempo real el flujo de activación a través de la capa oculta
y la capa de salida, con desglose aritmético y tabla de verdad sincronizada.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
from typing import Dict, Any


class PerceptronGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Simulador de Perceptrón Multicapa — Compuerta XOR")
        self.root.geometry("1020x680")
        self.root.minsize(920, 620)
        self.root.configure(bg="#121824")

        # Estado del perceptrón
        self.val_x = tk.IntVar(value=0)
        self.val_y = tk.IntVar(value=1)
        self.auto_running = False

        self._setup_styles()
        self._build_ui()
        self.update_network()

    def _setup_styles(self):
        self.colors = {
            "bg": "#121824",
            "panel_bg": "#1b2434",
            "card_bg": "#222f44",
            "accent": "#38bdf8",
            "accent_glow": "#0ea5e9",
            "active_green": "#22c55e",
            "inactive_red": "#ef4444",
            "inactive_node": "#334155",
            "text": "#f8fafc",
            "text_muted": "#94a3b8",
            "line_inactive": "#334155",
            "line_active": "#38bdf8",
            "border": "#2d3c54"
        }

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=self.colors["panel_bg"], height=70, padx=20, pady=12)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="Perceptrón Multicapa — Resolución del Operador XOR",
            font=("Helvetica", 18, "bold"),
            fg=self.colors["text"],
            bg=self.colors["panel_bg"]
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header,
            text="Teoría: X ⊕ Y = (X ∨ Y) ∧ (X̄ ∨ Ȳ)  •  Componer capas resuelve el problema no lineal",
            font=("Helvetica", 11),
            fg=self.colors["accent"],
            bg=self.colors["panel_bg"]
        )
        subtitle.pack(anchor="w")

        # Main container (dos columnas: izquierda gráfica, derecha controles y cálculos)
        main_frame = tk.Frame(self.root, bg=self.colors["bg"], padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Columna Izquierda: Canvas del Perceptrón
        left_col = tk.Frame(main_frame, bg=self.colors["panel_bg"], bd=1, relief=tk.SOLID)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        canvas_header = tk.Label(
            left_col,
            text="Topología de la Red Neuronal y Flujo de Señal",
            font=("Helvetica", 13, "bold"),
            fg=self.colors["text"],
            bg=self.colors["panel_bg"],
            pady=10
        )
        canvas_header.pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Canvas(
            left_col,
            bg="#0f172a",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Columna Derecha: Controles, Desglose Matemático y Tabla de Verdad
        right_col = tk.Frame(main_frame, bg=self.colors["bg"], width=420)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_col.pack_propagate(False)

        # 1. Panel de Controles
        ctrl_frame = tk.LabelFrame(
            right_col,
            text=" 🎛️ Control de Entradas ",
            font=("Helvetica", 11, "bold"),
            fg=self.colors["accent"],
            bg=self.colors["panel_bg"],
            padx=12,
            pady=10,
            bd=1,
            relief=tk.SOLID
        )
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        # Fila Entrada X
        row_x = tk.Frame(ctrl_frame, bg=self.colors["panel_bg"])
        row_x.pack(fill=tk.X, pady=4)
        tk.Label(row_x, text="Entrada X:", font=("Helvetica", 12, "bold"), fg=self.colors["text"], bg=self.colors["panel_bg"], width=10, anchor="w").pack(side=tk.LEFT)
        self.btn_x0 = tk.Button(row_x, text="0", font=("Helvetica", 11, "bold"), width=5, command=lambda: self.set_input('x', 0))
        self.btn_x0.pack(side=tk.LEFT, padx=4)
        self.btn_x1 = tk.Button(row_x, text="1", font=("Helvetica", 11, "bold"), width=5, command=lambda: self.set_input('x', 1))
        self.btn_x1.pack(side=tk.LEFT, padx=4)

        # Fila Entrada Y
        row_y = tk.Frame(ctrl_frame, bg=self.colors["panel_bg"])
        row_y.pack(fill=tk.X, pady=4)
        tk.Label(row_y, text="Entrada Y:", font=("Helvetica", 12, "bold"), fg=self.colors["text"], bg=self.colors["panel_bg"], width=10, anchor="w").pack(side=tk.LEFT)
        self.btn_y0 = tk.Button(row_y, text="0", font=("Helvetica", 11, "bold"), width=5, command=lambda: self.set_input('y', 0))
        self.btn_y0.pack(side=tk.LEFT, padx=4)
        self.btn_y1 = tk.Button(row_y, text="1", font=("Helvetica", 11, "bold"), width=5, command=lambda: self.set_input('y', 1))
        self.btn_y1.pack(side=tk.LEFT, padx=4)

        # Botón animación secuencial
        self.btn_cycle = tk.Button(
            ctrl_frame,
            text="▶ Demostrar los 4 casos (Auto)",
            font=("Helvetica", 10, "bold"),
            bg="#2563eb",
            fg="white",
            relief=tk.FLAT,
            pady=5,
            command=self.cycle_all_cases
        )
        self.btn_cycle.pack(fill=tk.X, pady=(8, 2))

        # 2. Panel de Cálculos Aritméticos
        math_frame = tk.LabelFrame(
            right_col,
            text=" 📐 Desglose Matemático en Tiempo Real ",
            font=("Helvetica", 11, "bold"),
            fg=self.colors["accent"],
            bg=self.colors["panel_bg"],
            padx=12,
            pady=8,
            bd=1,
            relief=tk.SOLID
        )
        math_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_math_h1 = tk.Label(math_frame, text="", font=("Courier", 10), justify=tk.LEFT, fg=self.colors["text"], bg=self.colors["card_bg"], padx=8, pady=6, anchor="w")
        self.lbl_math_h1.pack(fill=tk.X, pady=2)

        self.lbl_math_h2 = tk.Label(math_frame, text="", font=("Courier", 10), justify=tk.LEFT, fg=self.colors["text"], bg=self.colors["card_bg"], padx=8, pady=6, anchor="w")
        self.lbl_math_h2.pack(fill=tk.X, pady=2)

        self.lbl_math_out = tk.Label(math_frame, text="", font=("Courier", 10, "bold"), justify=tk.LEFT, fg=self.colors["text"], bg=self.colors["card_bg"], padx=8, pady=6, anchor="w")
        self.lbl_math_out.pack(fill=tk.X, pady=2)

        # 3. Tabla de Verdad
        table_frame = tk.LabelFrame(
            right_col,
            text=" 📊 Tabla de Verdad Sincronizada ",
            font=("Helvetica", 11, "bold"),
            fg=self.colors["accent"],
            bg=self.colors["panel_bg"],
            padx=8,
            pady=6,
            bd=1,
            relief=tk.SOLID
        )
        table_frame.pack(fill=tk.BOTH, expand=True)

        self.table_rows = []
        headers = ["X", "Y", "h1 (OR)", "h2 (NAND)", "Salida (XOR)"]
        header_row = tk.Frame(table_frame, bg="#0f172a")
        header_row.pack(fill=tk.X, pady=1)
        for h in headers:
            lbl = tk.Label(header_row, text=h, font=("Helvetica", 9, "bold"), fg=self.colors["accent"], bg="#0f172a", width=8)
            lbl.pack(side=tk.LEFT, expand=True, fill=tk.X)

        cases = [(0, 0, 0, 1, 0), (0, 1, 1, 1, 1), (1, 0, 1, 1, 1), (1, 1, 1, 0, 0)]
        for x, y, h1, h2, out in cases:
            row = tk.Frame(table_frame, bg=self.colors["card_bg"], padx=2, pady=3)
            row.pack(fill=tk.X, pady=1)
            vals = [str(x), str(y), str(h1), str(h2), f"{out} ✅"]
            labels = []
            for v in vals:
                l = tk.Label(row, text=v, font=("Helvetica", 9), fg=self.colors["text"], bg=self.colors["card_bg"], width=8)
                l.pack(side=tk.LEFT, expand=True, fill=tk.X)
                labels.append(l)
            self.table_rows.append({"key": (x, y), "frame": row, "labels": labels})

    def set_input(self, var_name: str, val: int):
        if var_name == 'x':
            self.val_x.set(val)
        else:
            self.val_y.set(val)
        self.update_network()

    def update_network(self):
        x = self.val_x.get()
        y = self.val_y.get()

        # Actualizar estilo de botones de control
        self.btn_x0.configure(bg=self.colors["accent"] if x == 0 else "#334155", fg="black" if x == 0 else "white")
        self.btn_x1.configure(bg=self.colors["accent"] if x == 1 else "#334155", fg="black" if x == 1 else "white")
        self.btn_y0.configure(bg=self.colors["accent"] if y == 0 else "#334155", fg="black" if y == 0 else "white")
        self.btn_y1.configure(bg=self.colors["accent"] if y == 1 else "#334155", fg="black" if y == 1 else "white")

        # Cálculos de la red
        # Capa Oculta
        z_h1 = 1.0 * x + 1.0 * y
        h1 = 1 if z_h1 >= 1.0 else 0

        z_h2 = -1.0 * x + -1.0 * y
        h2 = 1 if z_h2 >= -1.0 else 0

        # Capa de Salida
        z_out = 1.0 * h1 + 1.0 * h2
        out = 1 if z_out >= 2.0 else 0
        expected = x ^ y

        # Actualizar textos matemáticos
        self.lbl_math_h1.configure(
            text=f"h1 (OR): z1 = 1({x}) + 1({y}) = {z_h1:.0f}\n"
                 f"        ¿z1 >= 1.0? -> {'SÍ [DISPARA]' if h1 else 'NO'} => h1 = {h1}",
            fg=self.colors["active_green"] if h1 else self.colors["text_muted"]
        )

        self.lbl_math_h2.configure(
            text=f"h2 (NAND): z2 = -1({x}) - 1({y}) = {z_h2:.0f}\n"
                 f"          ¿z2 >= -1.0? -> {'SÍ [DISPARA]' if h2 else 'NO'} => h2 = {h2}",
            fg=self.colors["active_green"] if h2 else self.colors["text_muted"]
        )

        self.lbl_math_out.configure(
            text=f"Salida: z_out = 1({h1}) + 1({h2}) = {z_out:.0f}\n"
                 f"        ¿z_out >= 2.0? -> {'SÍ [DISPARA]' if out else 'NO'}\n"
                 f"        Resultado: {out} (XOR={expected}) -> {'✅ CORRECTO' if out == expected else '❌ ERROR'}",
            fg=self.colors["active_green"] if out else "#f87171"
        )

        # Resaltar fila en la tabla de verdad
        for item in self.table_rows:
            if item["key"] == (x, y):
                item["frame"].configure(bg="#1e3a8a")
                for lbl in item["labels"]:
                    lbl.configure(bg="#1e3a8a", fg="#38bdf8", font=("Helvetica", 9, "bold"))
            else:
                item["frame"].configure(bg=self.colors["card_bg"])
                for lbl in item["labels"]:
                    lbl.configure(bg=self.colors["card_bg"], fg=self.colors["text"], font=("Helvetica", 9))

        # Redibujar la red neuronal en el Canvas
        self._draw_diagram(x, y, z_h1, h1, z_h2, h2, z_out, out)

    def _draw_diagram(self, x, y, z_h1, h1, z_h2, h2, z_out, out):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        if w < 50:
            w, h = 540, 480

        # Coordenadas de los nodos
        # Entradas (columna 1)
        x_in = w * 0.16
        y_in1 = h * 0.32
        y_in2 = h * 0.68

        # Capa Oculta (columna 2)
        x_hid = w * 0.52
        y_hid1 = h * 0.30
        y_hid2 = h * 0.70

        # Salida (columna 3)
        x_out = w * 0.86
        y_out = h * 0.50

        r = 26  # Radio de nodos

        # 1. Dibujar conexiones (sinapsis) con pesos y activación
        # Entrada X -> h1 (peso +1)
        self._draw_synapse(x_in, y_in1, x_hid, y_hid1, weight="+1", active=(x == 1))
        # Entrada X -> h2 (peso -1)
        self._draw_synapse(x_in, y_in1, x_hid, y_hid2, weight="-1", active=(x == 1))
        # Entrada Y -> h1 (peso +1)
        self._draw_synapse(x_in, y_in2, x_hid, y_hid1, weight="+1", active=(y == 1))
        # Entrada Y -> h2 (peso -1)
        self._draw_synapse(x_in, y_in2, x_hid, y_hid2, weight="-1", active=(y == 1))

        # Capa oculta -> Salida
        # h1 -> Out (peso +1)
        self._draw_synapse(x_hid, y_hid1, x_out, y_out, weight="+1", active=(h1 == 1))
        # h2 -> Out (peso +1)
        self._draw_synapse(x_hid, y_hid2, x_out, y_out, weight="+1", active=(h2 == 1))

        # 2. Dibujar Nodos
        # Entradas
        self._draw_node(x_in, y_in1, r, text=f"X\n[{x}]", active=(x == 1), subtitle="Entrada")
        self._draw_node(x_in, y_in2, r, text=f"Y\n[{y}]", active=(y == 1), subtitle="Entrada")

        # Capa Oculta
        self._draw_node(
            x_hid, y_hid1, r,
            text="1",
            active=(h1 == 1),
            title="X ∨ Y",
            subtitle=f"z1={z_h1:.0f} (h1={h1})"
        )
        self._draw_node(
            x_hid, y_hid2, r,
            text="-1",
            active=(h2 == 1),
            title="X̄ ∨ Ȳ",
            subtitle=f"z2={z_h2:.0f} (h2={h2})"
        )

        # Salida
        self._draw_node(
            x_out, y_out, r + 4,
            text="2",
            active=(out == 1),
            title="X ⊕ Y",
            subtitle=f"z={z_out:.0f} (Salida={out})"
        )

        # Leyenda de Capas
        self.canvas.create_text(x_in, 22, text="Entradas", fill=self.colors["text_muted"], font=("Helvetica", 10, "bold"))
        self.canvas.create_text(x_hid, 22, text="Capa Oculta", fill=self.colors["accent"], font=("Helvetica", 10, "bold"))
        self.canvas.create_text(x_out, 22, text="Capa Salida", fill=self.colors["text_muted"], font=("Helvetica", 10, "bold"))

    def _draw_synapse(self, x1, y1, x2, y2, weight: str, active: bool):
        color = "#38bdf8" if active else "#334155"
        width = 3 if active else 1.5
        dash = () if active else (4, 4)

        # Línea con flecha
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, dash=dash, arrow=tk.LAST, arrowshape=(8, 10, 4))

        # Etiqueta de peso en el punto medio
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 - 8
        self.canvas.create_text(
            mx, my,
            text=weight,
            fill="#cbd5e1" if active else "#64748b",
            font=("Helvetica", 9, "bold")
        )

    def _draw_node(self, cx, cy, r, text: str, active: bool, title: str = "", subtitle: str = ""):
        # Colores de nodo
        if active:
            fill_color = "#164e63"
            outline_color = "#38bdf8"
            glow_color = "#0284c7"
            text_color = "#ffffff"
        else:
            fill_color = "#1e293b"
            outline_color = "#475569"
            glow_color = "#0f172a"
            text_color = "#94a3b8"

        # Halo exterior
        self.canvas.create_oval(cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4, outline=glow_color, width=2)
        # Círculo principal
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill_color, outline=outline_color, width=2.5)

        # Texto interior (umbral o valor)
        self.canvas.create_text(cx, cy, text=text, fill=text_color, font=("Helvetica", 11, "bold"), justify=tk.CENTER)

        # Título superior
        if title:
            self.canvas.create_text(cx, cy - r - 12, text=title, fill=self.colors["accent"], font=("Helvetica", 9, "bold"))

        # Subtítulo inferior
        if subtitle:
            self.canvas.create_text(cx, cy + r + 13, text=subtitle, fill="#cbd5e1", font=("Helvetica", 8))

    def cycle_all_cases(self):
        if self.auto_running:
            return
        self.auto_running = True
        cases = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 1)]

        def step(idx):
            if idx >= len(cases):
                self.auto_running = False
                return
            x, y = cases[idx]
            self.val_x.set(x)
            self.val_y.set(y)
            self.update_network()
            self.root.after(1200, lambda: step(idx + 1))

        step(0)


def launch():
    root = tk.Tk()
    app = PerceptronGUI(root)
    # Permite redibujar al redimensionar
    root.bind("<Configure>", lambda e: app.update_network() if e.widget == root else None)
    root.mainloop()


if __name__ == "__main__":
    launch()
