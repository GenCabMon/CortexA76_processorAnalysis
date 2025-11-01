import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Leer el CSV con las configuraciones top 10
df_top10 = pd.read_csv('top_10_configurations.csv')

# Función para parsear archivos stats.txt
def parse_stats_file(filename):
    stats = {}
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # Buscar simSeconds
            match_sim = re.search(r'simSeconds\s+([\d.]+)', content)
            if match_sim:
                stats['SimSeconds'] = float(match_sim.group(1))
            
            # Buscar CPI
            match_cpi = re.search(r'system\.cpu\.cpi\s+([\d.]+)', content)
            if match_cpi:
                stats['CPI'] = float(match_cpi.group(1))
            
            # Buscar IPC
            match_ipc = re.search(r'system\.cpu\.ipc\s+([\d.]+)', content)
            if match_ipc:
                stats['IPC'] = float(match_ipc.group(1))
    except FileNotFoundError:
        print(f"Archivo {filename} no encontrado")
    return stats

# Parsear los archivos de stats para 1stexec y 2ndexec
stats_1st = parse_stats_file('stats_rs_1stexec.txt')
stats_2nd = parse_stats_file('stats_rs_2ndexec.txt')

# Configuración por defecto
default_config = {
    'l1i_size': '64kB',
    'l1d_size': '64kB',
    'num_fu_intALU': 2,
    'btb_entries': 4096
}

# Función para convertir tamaño de cache a número
def cache_to_kb(size_str):
    if 'MB' in size_str:
        return float(size_str.replace('MB', '')) * 1024
    elif 'kB' in size_str:
        return float(size_str.replace('kB', ''))
    return float(size_str)

# Preparar datos de 1stexec y 2ndexec
exec_configs = []

# 1stexec (h264)
if stats_1st:
    config_1st = {
        'Name': '1stexec (H264)',
        'CPI': stats_1st.get('CPI', None),
        'IPC': stats_1st.get('IPC', None),
        'SimSeconds': stats_1st.get('SimSeconds', None),
        'ALU': 4,  # num_fu_intALU desde sa_results
        'BTB': 4096,
        'L1I': cache_to_kb('64kB'),
        'L1D': cache_to_kb('32kB')
    }
    exec_configs.append(config_1st)

# 2ndexec (jpeg2k)
if stats_2nd:
    config_2nd = {
        'Name': '2ndexec (JPEG2K)',
        'CPI': stats_2nd.get('CPI', None),
        'IPC': stats_2nd.get('IPC', None),
        'SimSeconds': stats_2nd.get('SimSeconds', None),
        'ALU': 20,  # num_fu_intALU desde sa_results
        'BTB': 4096,
        'L1I': cache_to_kb('32kB'),
        'L1D': cache_to_kb('32kB')
    }
    exec_configs.append(config_2nd)

# Convertir a DataFrame
df_exec = pd.DataFrame(exec_configs)

# Procesar datos del top 10
df_top10['L1I_KB'] = df_top10['L1I'].apply(cache_to_kb)
df_top10['L1D_KB'] = df_top10['L1D'].apply(cache_to_kb)

# Crear las gráficas individualmente

# 1. CPI vs ALUs
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(df_top10['ALU'], df_top10['CPI'], c='blue', s=100, alpha=0.6, label='Top 10 Configs')
if not df_exec.empty:
    ax1.scatter(df_exec['ALU'], df_exec['CPI'], c='red', s=150, marker='*', 
                label='SA Optimized', zorder=5)
    for idx, row in df_exec.iterrows():
        ax1.annotate(row['Name'], (row['ALU'], row['CPI']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
ax1.set_xlabel('Number of ALUs', fontsize=12, fontweight='bold')
ax1.set_ylabel('CPI (Cycles Per Instruction)', fontsize=12, fontweight='bold')
ax1.set_title('CPI vs ALUs', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
plt.tight_layout()
plt.savefig('cpi_vs_alus.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfica guardada: cpi_vs_alus.png")

# 2. SimSeconds vs ALUs
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(df_top10['ALU'], df_top10['SimSeconds'], c='blue', s=100, alpha=0.6, label='Top 10 Configs')
if not df_exec.empty:
    ax2.scatter(df_exec['ALU'], df_exec['SimSeconds'], c='red', s=150, marker='*', 
                label='SA Optimized', zorder=5)
    for idx, row in df_exec.iterrows():
        ax2.annotate(row['Name'], (row['ALU'], row['SimSeconds']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
ax2.set_xlabel('Number of ALUs', fontsize=12, fontweight='bold')
ax2.set_ylabel('Simulation Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('SimSeconds vs ALUs', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.savefig('simseconds_vs_alus.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfica guardada: simseconds_vs_alus.png")

# 3. IPC vs BTB
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.scatter(df_top10['BTB'], df_top10['IPC'], c='blue', s=100, alpha=0.6, label='Top 10 Configs')
if not df_exec.empty:
    ax3.scatter(df_exec['BTB'], df_exec['IPC'], c='red', s=150, marker='*', 
                label='SA Optimized', zorder=5)
    for idx, row in df_exec.iterrows():
        ax3.annotate(row['Name'], (row['BTB'], row['IPC']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
ax3.set_xlabel('BTB Entries', fontsize=12, fontweight='bold')
ax3.set_ylabel('IPC (Instructions Per Cycle)', fontsize=12, fontweight='bold')
ax3.set_title('IPC vs BTB Entries', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
plt.tight_layout()
plt.savefig('ipc_vs_btb.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfica guardada: ipc_vs_btb.png")

# 4. SimSeconds vs L1I
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.scatter(df_top10['L1I_KB'], df_top10['SimSeconds'], c='blue', s=100, alpha=0.6, label='Top 10 Configs')
if not df_exec.empty:
    ax4.scatter(df_exec['L1I'], df_exec['SimSeconds'], c='red', s=150, marker='*', 
                label='SA Optimized', zorder=5)
    for idx, row in df_exec.iterrows():
        ax4.annotate(row['Name'], (row['L1I'], row['SimSeconds']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
ax4.set_xlabel('L1I Cache Size (kB)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Simulation Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_title('SimSeconds vs L1I Cache Size', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()
plt.tight_layout()
plt.savefig('simseconds_vs_l1i.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfica guardada: simseconds_vs_l1i.png")

# 5. SimSeconds vs L1D
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.scatter(df_top10['L1D_KB'], df_top10['SimSeconds'], c='blue', s=100, alpha=0.6, label='Top 10 Configs')
if not df_exec.empty:
    ax5.scatter(df_exec['L1D'], df_exec['SimSeconds'], c='red', s=150, marker='*', 
                label='SA Optimized', zorder=5)
    for idx, row in df_exec.iterrows():
        ax5.annotate(row['Name'], (row['L1D'], row['SimSeconds']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
ax5.set_xlabel('L1D Cache Size (kB)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Simulation Time (seconds)', fontsize=12, fontweight='bold')
ax5.set_title('SimSeconds vs L1D Cache Size', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()
plt.tight_layout()
plt.savefig('simseconds_vs_l1d.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfica guardada: simseconds_vs_l1d.png")

# 6. Gráfica adicional: IPC vs CPI comparativo
fig6, ax6 = plt.subplots(figsize=(10, 6))
ax6.scatter(df_top10['CPI'], df_top10['IPC'], c='blue', s=100, alpha=0.6, label='Top 10 Configs')
if not df_exec.empty:
    ax6.scatter(df_exec['CPI'], df_exec['IPC'], c='red', s=150, marker='*', 
                label='SA Optimized', zorder=5)
    for idx, row in df_exec.iterrows():
        ax6.annotate(row['Name'], (row['CPI'], row['IPC']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
ax6.set_xlabel('CPI (Cycles Per Instruction)', fontsize=12, fontweight='bold')
ax6.set_ylabel('IPC (Instructions Per Cycle)', fontsize=12, fontweight='bold')
ax6.set_title('IPC vs CPI Relationship', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()
plt.tight_layout()
plt.savefig('ipc_vs_cpi.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfica guardada: ipc_vs_cpi.png")

print("\n✓ Todas las gráficas han sido guardadas exitosamente")

# Imprimir resumen de datos
print("\n=== RESUMEN DE DATOS ===")
print("\nTop 10 Configurations:")
print(df_top10[['Rank', 'ALU', 'BTB', 'L1I', 'L1D', 'CPI', 'IPC', 'SimSeconds']].to_string())

if not df_exec.empty:
    print("\n\nSimulated Annealing Optimized Configurations:")
    print(df_exec.to_string())