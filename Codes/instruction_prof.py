import re
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Carpeta donde están las simulaciones
sim_root = Path(".")
out_figs = Path("out_figs")
out_data = Path("out_data")
out_figs.mkdir(exist_ok=True)
out_data.mkdir(exist_ok=True)

# Expresiones regulares
issued_pattern = re.compile(r"system\.cpu\.statIssuedInstType_0::(\S+)\s+([0-9]+)")
branches_pattern = re.compile(r"system\.cpu\.executeStats0\.numBranches\s+(\d+)")

# Lista de claves que sabemos que son totales/aggregadas y no deben contarse en "Otras"
AGGREGATE_KEYS = {
    "total", "Total", "totalInsts", "total_insts", "TotalInsts", "All", "all", "total_instructions"
}

# Función para procesar un stats.txt y devolver summary y el diccionario 'otras'
def process_stats(stats_file, ignore_zero_values=True):
    counts = defaultdict(int)
    total_insts = 0
    num_branches = 0

    with open(stats_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = issued_pattern.search(line)
            if m:
                inst_type, count = m.group(1), int(m.group(2))
                counts[inst_type] += count
                total_insts += count
            b = branches_pattern.search(line)
            if b:
                num_branches = int(b.group(1))

    # Clasificación
    groups = {
        "Enteras": ["IntAlu", "IntMult", "IntDiv"],
        "Flotantes": [
            "FloatAdd", "FloatMult", "FloatDiv", "FloatCmp",
            "FloatCvt", "FloatMultAcc", "FloatMisc", "FloatSqrt"
        ],
        "Memoria": ["MemRead", "MemWrite"],
        "SIMD": [k for k in counts.keys() if k.startswith("Simd")],
    }

    summary = {}
    for category, keys in groups.items():
        summary[category] = sum(counts.get(k, 0) for k in keys)

    # Construir used_keys (incluir las keys de groups y las claves agregadas conocidas)
    used_keys = set()
    for keys in groups.values():
        used_keys.update(keys)
    used_keys.update({k for k in AGGREGATE_KEYS})

    # Filtrado para 'otras': excluir keys en used_keys y también claves vacías o que representen totales
    otras = {}
    for k, v in counts.items():
        if k in used_keys:
            continue
        # normalizar nombres de key que parezcan agregados (por seguridad)
        if k.lower().startswith("total") or "total" in k.lower():
            continue
        if ignore_zero_values and v == 0:
            # opcional: ignora claves con valor 0 para no ensuciar la lista
            continue
        otras[k] = v

    summary["Otras"] = sum(otras.values())
    summary["Branches"] = num_branches

    # Imprimir en consola las 'Otras' (lista ordenada por count desc)
    print(f"\n Archivo: {stats_file}")
    if otras:
        print(" Instrucciones en categoría 'Otras' (nombre: cuenta):")
        for k, v in sorted(otras.items(), key=lambda x: -x[1]):
            print(f"   {k}: {v}")
    else:
        print("✅ No hay instrucciones relevantes en categoría 'Otras' (o eran 0 y fueron ignoradas).")

    return summary, otras

# Recorrer directorios m5out_*
all_others = {}
for sim_dir in sim_root.glob("m5out_*"):
    if sim_dir.is_dir() and not sim_dir.name.startswith("m5out_gs"):
        stats_file = sim_dir / "stats.txt"
        if stats_file.exists():
            summary, otras = process_stats(stats_file)
            all_others[sim_dir.name] = otras

            # Graficar
            labels = list(summary.keys())
            values = list(summary.values())

            plt.figure(figsize=(8,5))
            plt.bar(labels, values)
            plt.xlabel("Clase de instrucción")
            plt.ylabel("Cantidad de instrucciones")
            plt.title(f"Instruction Class Profiling: {sim_dir.name}")
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

            # Guardar figura
            fig_path = out_figs / f"{sim_dir.name}.png"
            plt.savefig(fig_path)
            plt.close()
            print(f" Guardado: {fig_path}")
