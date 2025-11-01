#!/usr/bin/env python3
"""
compute_initial_temp.py
Calcula la temperatura inicial T0 para el algoritmo de recocido simulado
usando el método de desviación de aceptación.

T0 = k * σ
donde:
    k = -3 / ln(p)
    σ = desviación estándar de diferencias entre funciones objetivo
p = 0.85 (probabilidad de aceptación)

Se usan como "experimentaciones preliminares" las carpetas:
    m5out_*   (que NO empiecen por m5out_rs_)

Cada carpeta debe contener:
    - stats.txt
    - mcpat_output.txt
"""

import os
import re
import glob
import numpy as np

# --- Configuración ---
ACCEPTANCE_PROB = 0.85
STAT_KEYS = {
    "cpi": r"system\.cpu\.cpi\s+([0-9]+\.[0-9]+|[0-9]+)",
    "simSeconds": r"simSeconds\s+([0-9]+\.[0-9]+|[0-9]+)"
}

def parse_stats(path):
    data = {}
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    for k, pat in STAT_KEYS.items():
        m = re.search(pat, txt)
        data[k] = float(m.group(1)) if m else None
    return data

def parse_mcpat(path):
    data = {}
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    m_leak = re.search(r"Total Leakage\s*=\s*([0-9]+\.[0-9]+|[0-9]+)\s*W", txt)
    m_dyn = re.search(r"Runtime Dynamic\s*=\s*([0-9]+\.[0-9]+|[0-9]+)\s*W", txt)
    if m_leak: data["TotalLeakage"] = float(m_leak.group(1))
    if m_dyn:  data["RuntimeDynamic"] = float(m_dyn.group(1))
    return data

def compute_objective(stats, mcpat):
    """f = EDP + simSeconds + Energy"""
    if not all(k in mcpat for k in ("TotalLeakage","RuntimeDynamic")):
        return None
    if stats.get("cpi") is None or stats.get("simSeconds") is None:
        return None
    energy = (mcpat["TotalLeakage"] + mcpat["RuntimeDynamic"]) * stats["cpi"]
    edp = energy * stats["simSeconds"]
    return edp + stats["simSeconds"] + energy

def main():
    dirs = [d for d in glob.glob("m5out_*") if not os.path.basename(d).startswith("m5out_rs_")]
    objectives = []
    for d in sorted(dirs):
        sfile = os.path.join(d, "stats.txt")
        mfile = os.path.join(d, "mcpat_output.txt")
        if not os.path.exists(sfile) or not os.path.exists(mfile):
            continue
        s = parse_stats(sfile)
        m = parse_mcpat(mfile)
        f = compute_objective(s, m)
        if f is not None:
            objectives.append(f)

    if len(objectives) < 2:
        print("No hay suficientes simulaciones preliminares para calcular σ.")
        return

    # Diferencias entre valores consecutivos de la función objetivo
    diffs = np.diff(sorted(objectives))
    sigma = np.std(diffs)
    k = -3 / np.log(ACCEPTANCE_PROB)
    T0 = k * sigma * 300

    print(f"Simulaciones consideradas: {len(objectives)}")
    print(f"Desviación estándar (σ): {sigma:.6f}")
    print(f"k = {k:.6f}")
    print(f"Temperatura inicial (T0): {T0:.6f}")

if __name__ == "__main__":
    main()
