#!/usr/bin/env python3
"""
Analizador de resultados de gridsearch para gem5
Genera tablas y gráficas comparativas de las mejores configuraciones
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def guardar_nombres_configuraciones(df, top_n, filename="top_config_names.txt"):
    """
    Guarda los nombres de las mejores configuraciones (TOP N) en un archivo de texto.
    """
    top_df = df.nsmallest(top_n, 'SimSeconds').copy()

    with open(filename, 'w') as f:
        for idx, row in top_df.iterrows():
            f.write(f"{row['ConfigName']}\n")

    print(f"\nNombres de configuraciones TOP {top_n} guardados en: {filename}")

def extraer_metricas_stats(stats_file):
    """Extrae métricas específicas del archivo stats.txt"""
    metricas = {
        'SimSeconds': None,
        'CPI': None,
        'IPC': None,
        'BTBMispredicted': None,
        'BranchMispredicts': None
    }
    
    try:
        with open(stats_file, 'r') as f:
            for line in f:
                if 'simSeconds' in line and metricas['SimSeconds'] is None:
                    match = re.search(r'simSeconds\s+([\d.]+)', line)
                    if match:
                        metricas['SimSeconds'] = float(match.group(1))
                
                elif 'system.cpu.cpi' in line and metricas['CPI'] is None:
                    match = re.search(r'system\.cpu\.cpi\s+([\d.]+)', line)
                    if match:
                        metricas['CPI'] = float(match.group(1))
                
                elif 'system.cpu.ipc' in line and metricas['IPC'] is None:
                    match = re.search(r'system\.cpu\.ipc\s+([\d.]+)', line)
                    if match:
                        metricas['IPC'] = float(match.group(1))
                
                elif 'system.cpu.branchPred.BTBMispredicted' in line:
                    match = re.search(r'system\.cpu\.branchPred\.BTBMispredicted\s+([\d.]+)', line)
                    if match:
                        metricas['BTBMispredicted'] = float(match.group(1))
                
                elif 'system.cpu.commit.branchMispredicts' in line:
                    match = re.search(r'system\.cpu\.commit\.branchMispredicts\s+([\d.]+)', line)
                    if match:
                        metricas['BranchMispredicts'] = float(match.group(1))
    
    except FileNotFoundError:
        print(f"Advertencia: Archivo no encontrado: {stats_file}")
    
    return metricas

def analizar_gridsearch(csv_file='results_csv/gridsearch_results.csv', top_n=10):
    """Analiza los resultados del gridsearch y genera reportes"""
    
    # Leer CSV
    print(f"Leyendo resultados de {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Total de simulaciones: {len(df)}")
    
    # Crear nombres descriptivos para cada simulación
    df['ConfigName'] = df.apply(
        lambda row: f"{row['BranchPredictor']},{row['BP_Type']},{row['L1I']},{row['L1D']},{row['L2']},{row['L3']},{row['BTB']},{row['ALU']},{row['ROB']}", 
        axis=1
    )
    
    # Agregar métricas adicionales de stats.txt
    print("Extrayendo métricas adicionales de stats.txt...")
    
    btb_mispreds = []
    branch_mispreds = []
    
    for idx, row in df.iterrows():
        sim_name = f"gs_bp{row['BP_Type']}_L1I{row['L1I']}_L1D{row['L1D']}_L2{row['L2']}_L3{row['L3']}_BTB{row['BTB']}_ALU{row['ALU']}_ROB{row['ROB']}_W{row['PipelineWidth']}"
        stats_file = Path(f"m5out_{sim_name}/stats.txt")
        
        metricas = extraer_metricas_stats(stats_file)
        btb_mispreds.append(metricas['BTBMispredicted'])
        branch_mispreds.append(metricas['BranchMispredicts'])
    
    df['BTBMispredicted'] = btb_mispreds
    df['BranchMispredicts'] = branch_mispreds
    
    # Filtrar simulaciones válidas
    df_valido = df[df['SimSeconds'] > 0].copy()
    print(f"Simulaciones válidas: {len(df_valido)}")
    
    # Ordenar por SimSeconds (menor es mejor)
    df_sorted = df_valido.sort_values('SimSeconds').reset_index(drop=True)
    
    # Top N configuraciones para la tabla
    top_configs = df_sorted.head(top_n).copy()
    top_configs['Rank'] = range(1, len(top_configs) + 1)
    
    # Generar reporte
    print("\n" + "="*60)
    print(f"ESTADISTICAS: TOP {top_n} CONFIGURACIONES")
    print("="*60)
    
    # Métricas principales
    metricas = {
        'SimSeconds': 'menor',
        'CPI': 'menor',
        'IPC': 'mayor',
        'BTBMispredicted': 'menor',
        'BranchMispredicts': 'menor'
    }
    
    for metrica, criterio in metricas.items():
        if metrica in df_sorted.columns:
            if criterio == 'menor':
                mejor_idx = df_sorted[metrica].idxmin()
                peor_idx = df_sorted[metrica].idxmax()
            else:
                mejor_idx = df_sorted[metrica].idxmax()
                peor_idx = df_sorted[metrica].idxmin()
            
            mejor_val = df_sorted.loc[mejor_idx, metrica]
            peor_val = df_sorted.loc[peor_idx, metrica]
            
            if criterio == 'menor':
                mejora = ((peor_val - mejor_val) / peor_val * 100)
            else:
                mejora = ((mejor_val - peor_val) / peor_val * 100)
            
            print(f"\n{metrica}:")
            print(f"  Mejor:  {mejor_val:.6f}")
            print(f"  Peor:   {peor_val:.6f}")
            print(f"  Mejora: {mejora:.2f}%")
    
    # Tabla de ranking
    print("\n" + "="*100)
    print("RANKING COMPLETO (ordenado por SimSeconds)")
    print("="*100)
    print(f"{'#':<5} {'Configuracion':<80} {'SimSec':<12} {'CPI':<10} {'IPC':<10} {'BTBMisp':<12} {'BranchMisp':<12}")
    print("-" * 145)
    
    for idx, row in top_configs.iterrows():
        print(f"{row['Rank']:<5} {row['ConfigName']:<80} {row['SimSeconds']:<12.6f} {row['CPI']:<10.4f} "
              f"{row['IPC']:<10.4f} {row['BTBMispredicted']:<12.0f} "
              f"{row['BranchMispredicts']:<12.0f}")
    
    # Generar gráficas con TOP 10 de cada métrica
    generar_graficas(df_valido, top_n)
    
    # Guardar nombres de configuraciones en archivo
    guardar_nombres_configuraciones(df_valido, top_n)
    
    return top_configs

def generar_graficas(df_completo, top_n):
    """Genera 4 gráficas, cada una con el TOP 10 de su métrica correspondiente"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top {top_n} Configuraciones por Metrica Individual', 
                 fontsize=16, fontweight='bold')
    
    # Paleta de colores
    colors = sns.color_palette("viridis", top_n)
    
    # Gráfica 1: TOP 10 por SimSeconds (menor es mejor)
    ax1 = axes[0, 0]
    top_simsec = df_completo.nsmallest(top_n, 'SimSeconds').copy()
    top_simsec['ConfigName'] = [f"Sim_{i+1}" for i in range(len(top_simsec))]
    
    bars1 = ax1.barh(top_simsec['ConfigName'], top_simsec['SimSeconds'], color=colors)
    ax1.set_xlabel('Tiempo de Simulacion (segundos)', fontweight='bold')
    ax1.set_ylabel('Configuracion', fontweight='bold')
    ax1.set_title('SimSeconds - Top 10 (menor es mejor)', fontweight='bold')
    ax1.invert_yaxis()
    
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}s', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Gráfica 2: TOP 10 por CPI (menor es mejor)
    ax2 = axes[0, 1]
    top_cpi = df_completo.nsmallest(top_n, 'CPI').copy()
    top_cpi['ConfigName'] = [f"CPI_{i+1}" for i in range(len(top_cpi))]
    
    bars2 = ax2.barh(top_cpi['ConfigName'], top_cpi['CPI'], color=colors)
    ax2.set_xlabel('Ciclos por Instruccion', fontweight='bold')
    ax2.set_ylabel('Configuracion', fontweight='bold')
    ax2.set_title('CPI - Cycles Per Instruction Top 10 (menor es mejor)', fontweight='bold')
    ax2.invert_yaxis()
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Gráfica 3: TOP 10 por BTB Mispredictions (menor es mejor)
    ax3 = axes[1, 0]
    top_btb = df_completo.nsmallest(top_n, 'BTBMispredicted').copy()
    top_btb['ConfigName'] = [f"BTB_{i+1}" for i in range(len(top_btb))]
    
    bars3 = ax3.barh(top_btb['ConfigName'], top_btb['BTBMispredicted'], color=colors)
    ax3.set_xlabel('Numero de Fallos BTB', fontweight='bold')
    ax3.set_ylabel('Configuracion', fontweight='bold')
    ax3.set_title('BTB Mispredictions Top 10 (menor es mejor)', fontweight='bold')
    ax3.invert_yaxis()
    
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Gráfica 4: TOP 10 por Branch Mispredictions (menor es mejor)
    ax4 = axes[1, 1]
    top_branch = df_completo.nsmallest(top_n, 'BranchMispredicts').copy()
    top_branch['ConfigName'] = [f"Branch_{i+1}" for i in range(len(top_branch))]
    
    bars4 = ax4.barh(top_branch['ConfigName'], top_branch['BranchMispredicts'], color=colors)
    ax4.set_xlabel('Numero de Fallos de Prediccion', fontweight='bold')
    ax4.set_ylabel('Configuracion', fontweight='bold')
    ax4.set_title('Branch Mispredictions Top 10 (menor es mejor)', fontweight='bold')
    ax4.invert_yaxis()
    
    for i, bar in enumerate(bars4):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gridsearch_top_configs.png', dpi=300, bbox_inches='tight')
    print(f"\nGraficas guardadas en: gridsearch_top_configs.png")
    plt.show()

if __name__ == "__main__":
    # Analizar resultados
    top_configs = analizar_gridsearch(top_n=10)
    
    # Guardar reporte detallado
    top_configs.to_csv('top_10_configurations.csv', index=False)
    print(f"\nReporte detallado guardado en: top_10_configurations.csv")
