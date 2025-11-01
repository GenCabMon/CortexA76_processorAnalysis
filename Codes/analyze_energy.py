#!/usr/bin/env python3
"""
analyze_energy.py
- Busca carpetas m5out*.
- Para cada una lee:
    - stats.txt -> system.cpu.cpi, simSeconds, simInsts, hostSeconds, system.cpu.ipc
    - mcpat_output.txt -> Total Leakage, Runtime Dynamic (en W)
    - command.txt -> extrae --num_fu_intALU (opcional) y guarda comando.
- Calcula:
    Energy = (Total Leakage + Runtime Dynamic) * system.cpu.cpi
    E = Energy
    EDP = Energy * (simSeconds ** 1) * (system.cpu.cpi)   # Nota: ajustar si tienes otra definición.
    Objective = EDP + simSeconds + Energy   # según tu enunciado
- Genera:
    - Scatter 2D: Performance (simSeconds o IPC) vs Energy, con etiquetas.
    - Tabla CSV con valores y top3 por performance (menor simSeconds), energía (menor Energy) y EDP (menor EDP).
- Salidas: out_figs/energy_perf.png, out_data/energy_summary.csv
"""
import re, os, glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

STAT_KEYS = {
    "cpi": r"system\.cpu\.cpi\s+([0-9]+\.[0-9]+|[0-9]+)",
    "simSeconds": r"simSeconds\s+([0-9]+\.[0-9]+|[0-9]+)",
    "simInsts": r"simInsts\s+([0-9]+)",
    "ipc": r"system\.cpu\.ipc\s+([0-9]+\.[0-9]+|[0-9]+)",
    "hostSeconds": r"hostSeconds\s+([0-9]+\.[0-9]+|[0-9]+)"
}

def parse_stats(path):
    out = {}
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    for k, rx in STAT_KEYS.items():
        m = re.search(rx, txt)
        out[k] = float(m.group(1)) if m else None
    if out.get("simInsts") is not None:
        out["simInsts"] = int(out["simInsts"])
    return out

def parse_mcpat(path):
    """Busca 'Total Leakage = X W' y 'Runtime Dynamic = Y W' o variantes."""
    out = {}
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    m_leak = re.search(r"Total Leakage\s*=\s*([0-9]+\.[0-9]+|[0-9]+)\s*W", txt)
    if m_leak: out['total_leakage_W'] = float(m_leak.group(1))
    # Runtime Dynamic (top-level) commonly appears:
    m_rtd = re.search(r"Runtime Dynamic\s*=\s*([0-9]+\.[0-9]+|[0-9]+)\s*W", txt)
    if m_rtd: out['runtime_dynamic_W'] = float(m_rtd.group(1))
    # fallback: 'Runtime Dynamic = 2.33595 W' or 'Runtime Dynamic = 2.7238 W' etc.
    return out

def parse_command(path):
    if not os.path.exists(path):
        return {}
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    res = {}
    m = re.search(r"--num_fu_intALU\s*=?\s*([0-9]+)", txt)
    if not m:
        m = re.search(r"--num_fu_intALU\s+([0-9]+)", txt)
    if m:
        res['num_fu_intALU'] = int(m.group(1))
    res['command_line'] = txt.strip().splitlines()[-1] if txt.strip() else ""
    return res

def ensure_dirs():
    Path("out_figs").mkdir(exist_ok=True)
    Path("out_data").mkdir(exist_ok=True)

def main(pattern="m5out*"):
    ensure_dirs()
    rows = []
    for d in sorted(glob.glob(pattern)):
        stats_f = os.path.join(d, "stats.txt")
        mcpat_f = os.path.join(d, "mcpat_output.txt")
        cmd_f = os.path.join(d, "command.txt")
        if not os.path.isfile(stats_f):
            print("Skipping", d, "(no stats.txt)")
            continue
        s = parse_stats(stats_f)
        m = parse_mcpat(mcpat_f) if os.path.isfile(mcpat_f) else {}
        c = parse_command(cmd_f)
        total_leak = m.get("total_leakage_W")
        runtime_dyn = m.get("runtime_dynamic_W")
        cpi = s.get("cpi")
        simSeconds = s.get("simSeconds")
        ipc = s.get("ipc")
        # compute Energy if we have required pieces
        energy = None
        if total_leak is not None and runtime_dyn is not None and cpi is not None:
            energy = (total_leak + runtime_dyn) * cpi
        # compute EDP: here example EDP = Energy * simSeconds (classic E * t), but you used 'EDP + simSeconds + Energy' as objective.
        edp = None
        if energy is not None and simSeconds is not None:
            edp = energy * simSeconds
        objective = None
        if edp is not None and simSeconds is not None and energy is not None:
            objective = edp + simSeconds + energy
        rows.append({
            "sim_dir": os.path.basename(d),
            "cpi": cpi,
            "ipc": ipc,
            "simSeconds": simSeconds,
            "hostSeconds": s.get("hostSeconds"),
            "total_leakage_W": total_leak,
            "runtime_dynamic_W": runtime_dyn,
            "Energy": energy,
            "EDP": edp,
            "Objective": objective,
            "num_fu_intALU": c.get("num_fu_intALU"),
            "command_line": c.get("command_line","")
        })
    if not rows:
        print("No simulations found.")
        return
    df = pd.DataFrame(rows)
    df.to_csv("out_data/energy_summary.csv", index=False)

    # Scatter: performance vs energy. Use simSeconds (lower is better) — plot simSeconds on x, Energy on y.
    plt.figure(figsize=(6,5))
    have = df.dropna(subset=['simSeconds','Energy'])
    if not have.empty:
        plt.scatter(have['simSeconds'], have['Energy'])
        for i, r in have.iterrows():
            plt.annotate(r['sim_dir'], (r['simSeconds'], r['Energy']), fontsize=6, alpha=0.8)
        plt.xlabel("simSeconds (lower = faster)")
        plt.ylabel("Energy (W * CPI)")
        plt.title("Performance vs Energy (all simulations)")
        plt.tight_layout()
        plt.savefig("out_figs/energy_vs_perf.png")
        plt.close()

    # Identify top3: best performance (min simSeconds), best energy (min Energy), best EDP (min EDP)
    results = {}
    if not df['simSeconds'].isna().all():
        results['best_performance'] = df.sort_values('simSeconds').head(3)['sim_dir'].tolist()
    if not df['Energy'].isna().all():
        results['best_energy'] = df.sort_values('Energy').head(3)['sim_dir'].tolist()
    if not df['EDP'].isna().all():
        results['best_EDP'] = df.sort_values('EDP').head(3)['sim_dir'].tolist()

    # Save top lists
    pd.Series(results).to_json("out_data/top3_summary.json", indent=2)
    print("Saved out_data/energy_summary.csv, out_figs/energy_vs_perf.png, and out_data/top3_summary.json")
    print("Top3 summary:", results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="m5out*", help="glob pattern for simulation dirs")
    args = parser.parse_args()
    main(args.pattern)
