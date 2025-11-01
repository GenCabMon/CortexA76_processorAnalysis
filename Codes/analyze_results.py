#!/usr/bin/env python3
"""
Análisis y visualización de resultados de simulaciones gem5
Genera gráficas comparativas de cada grupo y comparaciones globales
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
import argparse

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class GEM5Analyzer:
    def __init__(self, csv_dir='results_csv'):
        self.csv_dir = Path(csv_dir)
        self.output_dir = Path('plots')
        self.output_dir.mkdir(exist_ok=True)
        
        # Colores para cada grupo
        self.colors = {
            'branch_predictors': 'steelblue',
            'l1_instruction_cache': 'green',
            'l1_data_cache': 'lightgreen',
            'l2_cache': 'coral',
            'l3_cache': 'orange',
            'btb_entries': 'purple',
            'integer_alu_units': 'red',
            'pipeline_widths': 'pink',
            'queue_sizes': 'brown',
            'optimizations': 'gold'
        }
    
    def load_group_csv(self, csv_file):
        """Cargar un CSV de grupo específico"""
        df = pd.read_csv(csv_file)
        df = df[df['ExitCode'] == 0]  # Solo simulaciones exitosas
        
        # Convertir a numérico
        numeric_cols = ['SimSeconds', 'HostSeconds', 'CPI', 'IPC', 
                        'BranchMispreds', 'BTBMispreds', 'PredictorMispreds',
                        'NumInsts', 'NumOps']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def list_available_csvs(self):
        """Listar todos los CSVs disponibles"""
        csv_files = sorted(self.csv_dir.glob('*.csv'))
        print(f"\n📁 CSVs disponibles en {self.csv_dir}/:\n")
        for i, csv_file in enumerate(csv_files, 1):
            line_count = sum(1 for _ in open(csv_file)) - 1
            print(f"   {i}. {csv_file.name} ({line_count} simulaciones)")
        return csv_files
    
    def plot_group_comparison(self, csv_file):
        """Generar gráficas de comparación para un grupo"""
        df = self.load_group_csv(csv_file)
        
        if len(df) == 0:
            print(f"⚠️  No hay datos válidos en {csv_file.name}")
            return
        
        group_name = csv_file.stem
        color = self.colors.get(group_name.split('_', 1)[1], 'steelblue')
        
        print(f"\n📊 Generando gráficas para: {group_name}")
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Análisis: {group_name.replace("_", " ").title()}', 
                     fontsize=16, fontweight='bold')
        
        # 1. CPI
        axes[0, 0].bar(range(len(df)), df['CPI'], color=color)
        axes[0, 0].set_title('CPI (Cycles Per Instruction)')
        axes[0, 0].set_ylabel('CPI (menor es mejor)')
        axes[0, 0].set_xlabel('Configuración')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(df['Name'], rotation=45, ha='right')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. IPC
        axes[0, 1].bar(range(len(df)), df['IPC'], color='green')
        axes[0, 1].set_title('IPC (Instructions Per Cycle)')
        axes[0, 1].set_ylabel('IPC (mayor es mejor)')
        axes[0, 1].set_xlabel('Configuración')
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels(df['Name'], rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. SimSeconds
        axes[0, 2].bar(range(len(df)), df['SimSeconds'], color='purple')
        axes[0, 2].set_title('Tiempo de Simulación')
        axes[0, 2].set_ylabel('Segundos (menor es mejor)')
        axes[0, 2].set_xlabel('Configuración')
        axes[0, 2].set_xticks(range(len(df)))
        axes[0, 2].set_xticklabels(df['Name'], rotation=45, ha='right')
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # 4. Branch Mispredictions
        axes[1, 0].bar(range(len(df)), df['BranchMispreds'], color='coral')
        axes[1, 0].set_title('Branch Mispredictions Totales')
        axes[1, 0].set_ylabel('Mispredictions')
        axes[1, 0].set_xlabel('Configuración')
        axes[1, 0].set_xticks(range(len(df)))
        axes[1, 0].set_xticklabels(df['Name'], rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 5. BTB vs Predictor Mispreds (stacked bar)
        x_pos = range(len(df))
        axes[1, 1].bar(x_pos, df['BTBMispreds'], label='BTB Misses', color='orange')
        axes[1, 1].bar(x_pos, df['PredictorMispreds'], bottom=df['BTBMispreds'], 
                       label='Predictor Misses', color='red')
        axes[1, 1].set_title('Desglose de Mispredictions')
        axes[1, 1].set_ylabel('Mispredictions')
        axes[1, 1].set_xlabel('Configuración')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels(df['Name'], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # 6. Speedup relativo (respecto al peor)
        baseline_simsec = df['SimSeconds'].max()
        speedup = baseline_simsec / df['SimSeconds']
        axes[1, 2].bar(range(len(df)), speedup, color='darkgreen')
        axes[1, 2].axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Baseline')
        axes[1, 2].set_title('Speedup Relativo')
        axes[1, 2].set_ylabel('Speedup (mayor es mejor)')
        axes[1, 2].set_xlabel('Configuración')
        axes[1, 2].set_xticks(range(len(df)))
        axes[1, 2].set_xticklabels(df['Name'], rotation=45, ha='right')
        axes[1, 2].legend()
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f'{group_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Guardado: {output_file}")
        plt.close()
        
        # Generar tabla de estadísticas
        self.generate_group_stats(df, group_name)
    
    def generate_group_stats(self, df, group_name):
        """Generar tabla de estadísticas del grupo"""
        stats_file = self.output_dir / f'{group_name}_stats.txt'
        
        with open(stats_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"ESTADÍSTICAS: {group_name.replace('_', ' ').upper()}\n")
            f.write(f"{'='*60}\n\n")
            
            # Mejor y peor por cada métrica
            metrics = {
                'CPI': ('menor', df['CPI']),
                'IPC': ('mayor', df['IPC']),
                'SimSeconds': ('menor', df['SimSeconds']),
                'BranchMispreds': ('menor', df['BranchMispreds'])
            }
            
            for metric, (criteria, values) in metrics.items():
                if criteria == 'menor':
                    best_idx = values.idxmin()
                    worst_idx = values.idxmax()
                else:
                    best_idx = values.idxmax()
                    worst_idx = values.idxmin()
                
                f.write(f"{metric}:\n")
                f.write(f"  ✅ Mejor:  {df.loc[best_idx, 'Name']}: {values[best_idx]:.6f}\n")
                f.write(f"  ❌ Peor:   {df.loc[worst_idx, 'Name']}: {values[worst_idx]:.6f}\n")
                
                if len(df) > 1:
                    improvement = abs((values[worst_idx] - values[best_idx]) / values[worst_idx] * 100)
                    f.write(f"  📈 Mejora: {improvement:.2f}%\n")
                f.write("\n")
            
            # Tabla completa ordenada por SimSeconds
            f.write(f"{'='*60}\n")
            f.write("RANKING COMPLETO (ordenado por SimSeconds)\n")
            f.write(f"{'='*60}\n\n")
            
            df_sorted = df.sort_values('SimSeconds')
            f.write(f"{'#':<4} {'Nombre':<25} {'SimSec':<12} {'CPI':<10} {'IPC':<10}\n")
            f.write(f"{'-'*60}\n")
            
            for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
                f.write(f"{i:<4} {row['Name']:<25} {row['SimSeconds']:<12.6f} "
                       f"{row['CPI']:<10.4f} {row['IPC']:<10.4f}\n")
        
        print(f"   📄 Estadísticas: {stats_file}")
    
    def plot_global_comparison(self):
        """Comparación global entre todos los grupos"""
        master_csv = self.csv_dir / '00_all_results.csv'
        
        if not master_csv.exists():
            print(f"⚠️  No se encontró {master_csv}")
            return
        
        print(f"\n📊 Generando comparación global...")
        df_all = self.load_group_csv(master_csv)
        
        # Top 10 configuraciones por diferentes métricas
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparación Global - Top 10 Configuraciones', 
                     fontsize=16, fontweight='bold')
        
        # Top 10 mejor CPI
        top_cpi = df_all.nsmallest(10, 'CPI')
        axes[0, 0].barh(range(len(top_cpi)), top_cpi['CPI'], color='steelblue')
        axes[0, 0].set_yticks(range(len(top_cpi)))
        axes[0, 0].set_yticklabels(top_cpi['Name'])
        axes[0, 0].set_xlabel('CPI (menor es mejor)')
        axes[0, 0].set_title('Top 10 - Mejor CPI')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Top 10 mejor IPC
        top_ipc = df_all.nlargest(10, 'IPC')
        axes[0, 1].barh(range(len(top_ipc)), top_ipc['IPC'], color='green')
        axes[0, 1].set_yticks(range(len(top_ipc)))
        axes[0, 1].set_yticklabels(top_ipc['Name'])
        axes[0, 1].set_xlabel('IPC (mayor es mejor)')
        axes[0, 1].set_title('Top 10 - Mejor IPC')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Top 10 menor SimSeconds
        top_simsec = df_all.nsmallest(10, 'SimSeconds')
        axes[1, 0].barh(range(len(top_simsec)), top_simsec['SimSeconds'], color='purple')
        axes[1, 0].set_yticks(range(len(top_simsec)))
        axes[1, 0].set_yticklabels(top_simsec['Name'])
        axes[1, 0].set_xlabel('SimSeconds (menor es mejor)')
        axes[1, 0].set_title('Top 10 - Menor Tiempo de Simulación')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Top 10 menos Branch Mispredictions
        top_branch = df_all.nsmallest(10, 'BranchMispreds')
        axes[1, 1].barh(range(len(top_branch)), top_branch['BranchMispreds'], color='coral')
        axes[1, 1].set_yticks(range(len(top_branch)))
        axes[1, 1].set_yticklabels(top_branch['Name'])
        axes[1, 1].set_xlabel('Branch Mispredictions')
        axes[1, 1].set_title('Top 10 - Menos Branch Mispredictions')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'global_top10_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Guardado: {output_file}")
        plt.close()
        
        # Scatter plot: CPI vs Branch Mispredictions
        self.plot_scatter_analysis(df_all)
    
    def plot_scatter_analysis(self, df_all):
        """Análisis de correlación entre métricas"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Análisis de Correlación', fontsize=16, fontweight='bold')
        
        # CPI vs Branch Mispredictions
        axes[0].scatter(df_all['BranchMispreds'], df_all['CPI'], alpha=0.6, s=100)
        axes[0].set_xlabel('Branch Mispredictions')
        axes[0].set_ylabel('CPI')
        axes[0].set_title('CPI vs Branch Mispredictions')
        axes[0].grid(True, alpha=0.3)
        
        # Agregar línea de tendencia
        z = np.polyfit(df_all['BranchMispreds'], df_all['CPI'], 1)
        p = np.poly1d(z)
        axes[0].plot(df_all['BranchMispreds'], p(df_all['BranchMispreds']), 
                     "r--", alpha=0.8, linewidth=2, label='Tendencia')
        axes[0].legend()
        
        # SimSeconds vs CPI
        axes[1].scatter(df_all['CPI'], df_all['SimSeconds'], alpha=0.6, s=100, color='green')
        axes[1].set_xlabel('CPI')
        axes[1].set_ylabel('SimSeconds')
        axes[1].set_title('SimSeconds vs CPI')
        axes[1].grid(True, alpha=0.3)
        
        # Línea de tendencia
        z2 = np.polyfit(df_all['CPI'], df_all['SimSeconds'], 1)
        p2 = np.poly1d(z2)
        axes[1].plot(df_all['CPI'], p2(df_all['CPI']), 
                     "r--", alpha=0.8, linewidth=2, label='Tendencia')
        axes[1].legend()
        
        plt.tight_layout()
        output_file = self.output_dir / 'correlation_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Guardado: {output_file}")
        plt.close()
    
    def analyze_all(self):
        """Analizar todos los grupos"""
        csv_files = sorted(self.csv_dir.glob('[0-9]*.csv'))
        
        if not csv_files:
            print(f"⚠️  No se encontraron CSVs en {self.csv_dir}/")
            return
        
        print(f"\n🔍 Analizando {len(csv_files)} grupos de simulaciones...")
        
        for csv_file in csv_files:
            if csv_file.name.startswith('00_'):
                continue  # Skip master file
            self.plot_group_comparison(csv_file)
        
        # Comparación global
        self.plot_global_comparison()
        
        print(f"\n✨ Análisis completo. Gráficas guardadas en: {self.output_dir}/")
    
    def analyze_specific(self, group_number):
        """Analizar un grupo específico"""
        pattern = f'{group_number:02d}_*.csv'
        csv_files = list(self.csv_dir.glob(pattern))
        
        if not csv_files:
            print(f"⚠️  No se encontró CSV para grupo {group_number}")
            return
        
        self.plot_group_comparison(csv_files[0])
        print(f"\n✨ Análisis del grupo {group_number} completado")

def main():
    parser = argparse.ArgumentParser(
        description='Analizar resultados de simulaciones gem5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --all              # Analizar todos los grupos
  %(prog)s --group 1          # Analizar solo branch predictors
  %(prog)s --group 2          # Analizar solo L1I cache
  %(prog)s --list             # Listar CSVs disponibles
  %(prog)s --global           # Solo comparación global
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Analizar todos los grupos')
    parser.add_argument('--group', type=int, metavar='N',
                       help='Analizar un grupo específico (1-10)')
    parser.add_argument('--list', action='store_true',
                       help='Listar CSVs disponibles')
    parser.add_argument('--global', action='store_true', dest='global_only',
                       help='Solo generar comparación global')
    parser.add_argument('--csv-dir', default='results_csv',
                       help='Directorio con los CSVs (default: results_csv)')
    
    args = parser.parse_args()
    
    analyzer = GEM5Analyzer(csv_dir=args.csv_dir)
    
    if args.list:
        analyzer.list_available_csvs()
    elif args.all:
        analyzer.analyze_all()
    elif args.group:
        analyzer.analyze_specific(args.group)
    elif args.global_only:
        analyzer.plot_global_comparison()
    else:
        parser.print_help()
        print("\n💡 Usa --all para analizar todos los grupos")

if __name__ == '__main__':
    main()
