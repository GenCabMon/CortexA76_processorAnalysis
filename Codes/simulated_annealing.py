#!/usr/bin/env python3
"""
simulated_annealing.py
Implementación de Simulated Annealing para optimizar configuración del procesador
Función objetivo: f(x) = EDP + Energy + SimSeconds
VERSIÓN CON THREADS PARA EVALUACIÓN PARALELA DE VECINOS
"""

import os
import re
import random
import math
import subprocess
import time
from pathlib import Path
from threading import Thread, Lock
from queue import Queue

# ============================================
# CONFIGURACIÓN DEL ALGORITMO
# ============================================
ALPHA = 0.85  # Factor de enfriamiento geométrico
T_MIN = 0.01  # Temperatura mínima
MAX_ITERATIONS = 70  # Máximo de iteraciones para test
MIN_NEIGHBORS = 6  # Mínimo de vecinos a generar
MAX_THREADS = 6  # Número máximo de threads paralelos
FUNC_SCALER = 100  # Escalador para función objetivo
SIM_SEC_MULTIPLIER = 1000  # Multiplicador para SimSeconds en función objetivo: h264 = 1000, jpeg2k = 10

# Rutas
GEM5_BIN = "./build/ARM/gem5.fast"
CONFIG_SCRIPT = "scripts/CortexA76_scripts_gem5/CortexA76.py"
BENCHMARK = "workloads/h264_dec/h264_dec"
BENCHMARK_OPTS = "-i workloads/h264_dec/h264dec_testfile.264 -o h264_out.yuv"
MCPAT_EXEC = "../mcpat/mcpat"
MCPAT_SCRIPT = "scripts/McPAT/gem5toMcPAT_cortexA76.py"
MCPAT_TEMPLATE = "scripts/McPAT/ARM_A76_2.1GHz.xml"

# Lock para imprimir de forma segura
print_lock = Lock()

# ============================================
# ESPACIO DE BÚSQUEDA
# ============================================
PARAM_SPACE = {
    # Memory Options (strings - tamaños de caché)
    'l1i_size': ['32kB', '64kB', '128kB', '256kB'],
    'l1d_size': ['32kB', '64kB', '128kB', '256kB'],
    'l2_size': ['128kB', '256kB', '512kB', '1MB', '2MB'],
    'l3_size': ['1MB', '2MB', '4MB', '8MB'],
    
    # Memory Options (integers)
    'l1i_assoc': (1, 32),
    'l1i_lat': (1, 32),
    'l1d_assoc': (1, 32),
    'l1d_lat': (1, 32),
    'l2_assoc': (1, 32),
    'l2_lat': (1, 32),
    'l3_assoc': (1, 32),
    'l3_lat': (1, 32),
    
    # CPU Options - widths
    'fetch_width': (1, 16),
    'decode_width': (1, 16),
    'rename_width': (1, 16),
    'commit_width': (1, 16),
    'dispatch_width': (1, 16),
    'issue_width': (1, 16),
    'wb_width': (1, 16),
    
    # Queue entries
    'fb_entries': (1, 512),
    'fq_entries': (1, 512),
    'iq_entries': (1, 512),
    'rob_entries': (1, 512),
    'lq_entries': (1, 512),
    'sq_entries': (1, 512),
    'btb_entries': [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    'ras_entries': (1, 512),
    
    # Functional Units
    'num_fu_cmp': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
    'num_fu_intALU': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
    'num_fu_intDIVMUL': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
    'num_fu_FP_SIMD_ALU': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
    'num_fu_read': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
    'num_fu_write': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
    
    # Branch Predictor
    'branch_predictor_type': (0, 10)
}

# Valores default
DEFAULT_CONFIG = {
    'l1i_size': '32kB', 'l1i_assoc': 2, 'l1i_lat': 2,
    'l1d_size': '32kB', 'l1d_assoc': 2, 'l1d_lat': 2,
    'l2_size': '128kB', 'l2_assoc': 8, 'l2_lat': 2,
    'l3_size': '2MB', 'l3_assoc': 16, 'l3_lat': 4,
    'fetch_width': 7, 'decode_width': 3, 'rename_width': 6,
    'commit_width': 7, 'dispatch_width': 7, 'issue_width': 3,
    'wb_width': 5, 'fb_entries': 16, 'fq_entries': 459,
    'iq_entries': 16, 'rob_entries': 72, 'lq_entries': 16,
    'sq_entries': 68, 'btb_entries': 4096, 'ras_entries': 319,
    'num_fu_cmp': 28, 'num_fu_intALU': 20, 'num_fu_intDIVMUL': 2,
    'num_fu_FP_SIMD_ALU': 1, 'num_fu_read': 32, 'num_fu_write': 4,
    'branch_predictor_type': 9
}

# ============================================
# FUNCIONES AUXILIARES
# ============================================
def safe_print(*args, **kwargs):
    """Imprime de forma thread-safe"""
    with print_lock:
        print(*args, **kwargs)

def cache_size_to_bytes(size_str):
    """Convierte '64kB' a bytes para comparación"""
    if size_str.endswith('kB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    return 0

def validate_config(config):
    """Valida que la configuración cumpla las restricciones"""
    # Restricción de jerarquía de caché
    l1i = cache_size_to_bytes(config['l1i_size'])
    l1d = cache_size_to_bytes(config['l1d_size'])
    l2 = cache_size_to_bytes(config['l2_size'])
    l3 = cache_size_to_bytes(config['l3_size'])
    
    if not (l1i < l2 and l1d < l2 and l2 < l3):
        return False
    
    # Branch predictor entre 0-10
    if not (0 <= config['branch_predictor_type'] <= 10):
        return False
    
    # Todos los valores deben ser positivos
    for key, value in config.items():
        if isinstance(value, int) and value <= 0:
            return False
    
    return True

def parse_stats_file(stats_path):
    """Extrae métricas de stats.txt"""
    data = {}
    try:
        with open(stats_path, 'r') as f:
            text = f.read()
        
        patterns = {
            'cpi': r'system\.cpu\.cpi\s+([0-9]+\.[0-9]+)',
            'simSeconds': r'simSeconds\s+([0-9]+\.[0-9]+)',
            'hostSeconds': r'hostSeconds\s+([0-9]+\.[0-9]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = float(match.group(1))
    except Exception as e:
        safe_print(f"Error parsing stats.txt: {e}")
    
    return data

def parse_mcpat_output(mcpat_path):
    """Extrae métricas de mcpat_output.txt"""
    data = {}
    try:
        with open(mcpat_path, 'r') as f:
            text = f.read()
        
        patterns = {
            'TotalLeakage': r'Total Leakage\s*=\s*([0-9]+\.[0-9]+)\s*W',
            'RuntimeDynamic': r'Runtime Dynamic\s*=\s*([0-9]+\.[0-9]+)\s*W'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = float(match.group(1))
    except Exception as e:
        safe_print(f"Error parsing mcpat_output.txt: {e}")
    
    return data

def compute_objective(stats, mcpat):
    """
    Calcula la función objetivo:
    f(x) = EDP + Energy + SimSeconds
    donde:
        Energy = (Total Leakage + Runtime Dynamic) * CPI
        EDP = Energy * SimSeconds
    """
    try:
        if 'cpi' not in stats or 'simSeconds' not in stats:
            return float('inf')
        if 'TotalLeakage' not in mcpat or 'RuntimeDynamic' not in mcpat:
            return float('inf')
        
        cpi = stats['cpi']
        sim_seconds = stats['simSeconds']
        total_leakage = mcpat['TotalLeakage']
        runtime_dynamic = mcpat['RuntimeDynamic']
        
        energy = (total_leakage + runtime_dynamic) * cpi
        edp = energy * cpi
        
        return (edp + energy*1.5 + SIM_SEC_MULTIPLIER*sim_seconds) * FUNC_SCALER
    except:
        return float('inf')

def run_simulation(config, iteration, is_neighbor=False, neighbor_id=None):
    """Ejecuta una simulación de gem5 con la configuración dada"""
    # Crear nombre de directorio
    prefix = "m5out_rs_neighbor" if is_neighbor else "m5out_rs"
    neighbor_suffix = f"_n{neighbor_id}" if neighbor_id is not None else ""
    outdir = f"{prefix}_iter{iteration}{neighbor_suffix}_{int(time.time() * 1000)}"
    
    # Construir comando gem5
    params = []
    for key, value in config.items():
        params.append(f"--{key}={value}")
    
    cmd = [
        GEM5_BIN, '-d', outdir,
        CONFIG_SCRIPT,
        '-c', BENCHMARK,
        '-o', BENCHMARK_OPTS
    ] + params
    
    safe_print(f"\n[SIMULATION] Ejecutando: {outdir}")
    safe_print("[CMD-GEM5] " + " ".join(cmd))
    
    error_reason = None
    
    # Ejecutar gem5
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            error_reason = f"gem5 failed with code {result.returncode}"
            safe_print(f"[ERROR] {error_reason}")
            return None, float('inf'), error_reason
        
        # Verificar que se generaron los archivos
        stats_file = Path(outdir) / 'stats.txt'
        config_file = Path(outdir) / 'config.json'
        
        if not stats_file.exists() or not config_file.exists():
            error_reason = "Missing output files (stats.txt or config.json)"
            safe_print(f"[ERROR] {error_reason}")
            return None, float('inf'), error_reason
        
        # Ejecutar McPAT
        safe_print(f"[MCPAT] Generando análisis de energía...")
        mcpat_config = Path(outdir) / 'config.xml'
        mcpat_output = Path(outdir) / 'mcpat_output.txt'
        
        # Crear nombre único por vecino
        unique_xml_name = f"config_iter{iteration}_n{neighbor_id}.xml"

        # Ejecutar el script gem5toMcPAT indicando salida única
        unique_xml = f"config_iter{iteration}_n{neighbor_id}.xml"

        mcpat_cmd = [
            'python3', MCPAT_SCRIPT,
            str(stats_file), str(config_file), MCPAT_TEMPLATE,
            "--output", unique_xml
        ]
        subprocess.run(mcpat_cmd, capture_output=True)

        mcpat_config = Path(outdir) / unique_xml
        subprocess.run(["mv", unique_xml, str(mcpat_config)])
        
        # Verificar que se generó XML válido
        if not mcpat_config.exists() or mcpat_config.stat().st_size == 0:
            error_reason = "No se generó XML válido para McPAT"
            safe_print(f"[ERROR] {error_reason}")
            return None, float('inf'), error_reason
        
        # Ejecutar McPAT
        mcpat_result = subprocess.run(
            [MCPAT_EXEC, '-infile', str(mcpat_config), '-print_level', '1'],
            capture_output=True, text=True
        )
        
        with open(mcpat_output, 'w') as f:
            f.write(mcpat_result.stdout)
            f.write(mcpat_result.stderr)
        
        # Verificar si McPAT tuvo éxito
        if "XML Parsing error" in mcpat_result.stdout or "error" in mcpat_result.stderr.lower():
            error_reason = "McPAT failed - XML Parsing error"
            safe_print(f"[ERROR] {error_reason}")
            return None, float('inf'), error_reason

        # Parsear resultados
        stats = parse_stats_file(stats_file)
        mcpat = parse_mcpat_output(mcpat_output)
        
        objective = compute_objective(stats, mcpat)
        
        if objective == float('inf'):
            error_reason = "Objective computation failed - missing metrics"
        
        safe_print(f"[RESULT] Objective = {objective:.6f}")
        if objective < float('inf'):
            safe_print(f"         CPI={stats.get('cpi', 0):.4f}, "
                  f"SimSec={stats.get('simSeconds', 0):.6f}, "
                  f"Energy={(mcpat.get('TotalLeakage',0) + mcpat.get('RuntimeDynamic',0)) * stats.get('cpi',1):.6f}")
        
        return outdir, objective, error_reason
        
    except subprocess.TimeoutExpired:
        error_reason = "Simulation timeout (>3600s)"
        safe_print(f"[ERROR] {error_reason}")
        return None, float('inf'), error_reason
    except Exception as e:
        error_reason = f"Exception: {str(e)}"
        safe_print(f"[ERROR] {error_reason}")
        return None, float('inf'), error_reason

def generate_neighbor(current_config):
    """Genera un vecino válido aplicando operadores de movimiento"""
    max_attempts = 100
    
    for _ in range(max_attempts):
        neighbor = current_config.copy()
        
        # Seleccionar operador: 70% modificación, 30% intercambio
        if random.random() < 0.7:
            # Operador de MODIFICACIÓN
            safe_print("Modificación de un parámetro")
            param = random.choice(list(PARAM_SPACE.keys()))
            space = PARAM_SPACE[param]
            
            if isinstance(space, list):
                # Parámetro con lista de valores discretos
                neighbor[param] = random.choice(space)
            elif isinstance(space, tuple):
                # Parámetro con rango continuo
                min_val, max_val = space
                neighbor[param] = random.randint(min_val, max_val)
        else:
            # Operador de INTERCAMBIO
            # Seleccionar dos parámetros int
            safe_print("Intercambio de parámetros enteros")
            int_params = [k for k, v in PARAM_SPACE.items() 
                         if isinstance(v, tuple) and isinstance(current_config[k], int)]
            
            if len(int_params) >= 2:
                param1, param2 = random.sample(int_params, 2)
                neighbor[param1], neighbor[param2] = neighbor[param2], neighbor[param1]
        
        # Validar configuración
        if validate_config(neighbor):
            return neighbor
    
    # Si no se pudo generar vecino válido, retornar configuración actual
    return current_config

def worker_thread(work_queue, results_queue, iteration):
    """Thread worker que procesa vecinos de la cola"""
    while True:
        try:
            item = work_queue.get()
            if item is None:  # Señal de terminación
                work_queue.task_done()
                break
            
            neighbor_id, neighbor_config = item
            neighbor_dir, neighbor_obj, error_reason = run_simulation(
                neighbor_config, iteration, is_neighbor=True, neighbor_id=neighbor_id
            )
            
            results_queue.put({
                'id': neighbor_id,
                'config': neighbor_config,
                'directory': neighbor_dir,
                'objective': neighbor_obj,
                'error': error_reason
            })
            
            work_queue.task_done()
        except Exception as e:
            safe_print(f"[THREAD ERROR] Exception in worker thread: {e}")
            work_queue.task_done()

def evaluate_neighbors_parallel(neighbors, iteration, max_threads=MAX_THREADS):
    """Evalúa múltiples vecinos en paralelo usando threads"""
    work_queue = Queue()
    results_queue = Queue()
    
    # Crear threads
    threads = []
    num_threads = min(max_threads, len(neighbors))
    for _ in range(num_threads):
        t = Thread(target=worker_thread, args=(work_queue, results_queue, iteration))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Encolar trabajos
    for i, neighbor in enumerate(neighbors):
        work_queue.put((i, neighbor))
    
    # Señal de terminación para cada thread
    for _ in threads:
        work_queue.put(None)
    
    # Esperar a que terminen todos los trabajos
    work_queue.join()
    
    # Esperar a que todos los threads terminen completamente
    for t in threads:
        t.join(timeout=10)
    
    # Recoger resultados
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    # Ordenar por ID para mantener consistencia
    results.sort(key=lambda x: x['id'])
    
    return results

def select_neighbor(neighbors, objectives, current_obj, temperature):
    """
    Selecciona el mejor vecino o acepta uno peor con probabilidad de Boltzmann
    """
    if not neighbors:
        return None, float('inf')
    
    # Ordenar vecinos por objetivo (menor es mejor)
    sorted_neighbors = sorted(zip(neighbors, objectives), key=lambda x: x[1])
    
    best_neighbor, best_obj = sorted_neighbors[0]
    
    # Si hay mejora, aceptar inmediatamente
    if best_obj < current_obj:
        safe_print(f"[ACCEPT] Mejora encontrada: {best_obj:.6f} < {current_obj:.6f}")
        return best_neighbor, best_obj
    
    # Si no hay mejora, evaluar con probabilidad de Boltzmann
    safe_print(f"[EVALUATE] No hay mejora directa, evaluando probabilidades...")
    
    for neighbor, obj in sorted_neighbors:
        delta = obj - current_obj
        probability = math.exp(-delta / temperature)
        rand_val = random.random()
        
        safe_print(f"  Δ={delta:.4f}, P(accept)={probability:.4f}, rand={rand_val:.4f}", end="")
        
        if rand_val < probability:
            safe_print(" → ACEPTA")
            return neighbor, obj
        else:
            safe_print(" → RECHAZA")
    
    # Si ninguno fue aceptado, mantener solución actual
    safe_print(f"[KEEP] Manteniendo solución actual")
    return None, current_obj

import shutil
from pathlib import Path

def cleanup_iteration_dirs(iteration):
    pattern = f"m5out_rs_iter{iteration}"
    for path in Path(".").glob(f"{pattern}*"):
        try:
            shutil.rmtree(path)
            safe_print(f"[CLEANUP] Eliminado: {path}")
        except Exception as e:
            safe_print(f"[CLEANUP ERROR] No se pudo eliminar {path}: {e}")

def simulated_annealing(initial_config, initial_temp):
    """
    Algoritmo principal de Simulated Annealing
    """
    print("\n" + "="*60)
    print("SIMULATED ANNEALING - OPTIMIZACIÓN DE PROCESADOR")
    print("VERSIÓN PARALELA CON THREADS")
    print("="*60)
    print(f"Temperatura inicial: {initial_temp:.6f}")
    print(f"Factor de enfriamiento: {ALPHA}")
    print(f"Temperatura mínima: {T_MIN}")
    print(f"Máximo de iteraciones: {MAX_ITERATIONS}")
    print(f"Threads paralelos: {MAX_THREADS}")
    print("="*60 + "\n")
    
    # Estado inicial
    current_config = initial_config
    current_temp = initial_temp
    iteration = 0
    
    # Evaluar solución inicial
    print(f"\n{'='*60}")
    print(f"ITERACIÓN 0 - SOLUCIÓN INICIAL")
    print(f"{'='*60}")
    
    current_dir, current_obj, error_reason = run_simulation(current_config, iteration)
    
    if current_obj == float('inf'):
        print("[ERROR] No se pudo evaluar la solución inicial")
        print(f"[ERROR] Razón: {error_reason}")
        print("[DEBUG] Revisando archivos generados...")
        if current_dir:
            print(f"[DEBUG] Directorio: {current_dir}")
            stats_file = Path(current_dir) / 'stats.txt'
            mcpat_file = Path(current_dir) / 'mcpat_output.txt'
            if stats_file.exists():
                print(f"[DEBUG] stats.txt existe")
            if mcpat_file.exists():
                print(f"[DEBUG] mcpat_output.txt existe")
                # Ver contenido
                with open(mcpat_file) as f:
                    content = f.read()
                    if "XML Parsing error" in content:
                        print("[ERROR] McPAT tiene error de XML parsing")
                    print(f"[DEBUG] Primeras líneas de mcpat_output.txt:")
                    print(content[:500])
        return current_config, current_obj, []
    
    best_config = current_config
    best_obj = current_obj
    best_dir = current_dir
    
    print(f"\n[INITIAL] Objetivo inicial: {current_obj:.6f}")
    
    # Historia
    history = [{
        'iteration': 0,
        'temperature': current_temp,
        'objective': current_obj,
        'best_objective': best_obj,
        'directory': current_dir
    }]
    
    # Loop principal
    while current_temp > T_MIN and iteration < MAX_ITERATIONS:
        iteration += 1
        
        print(f"\n{'='*60}")
        print(f"ITERACIÓN {iteration}")
        print(f"{'='*60}")
        print(f"Temperatura actual: {current_temp:.6f}")
        print(f"Mejor objetivo hasta ahora: {best_obj:.6f}")
        
        # Generar vecinos
        print(f"\n[NEIGHBORS] Generando {MIN_NEIGHBORS} vecinos...")
        neighbors = []
        for i in range(MIN_NEIGHBORS):
            neighbor = generate_neighbor(current_config)
            neighbors.append(neighbor)
        
        # Evaluar vecinos en paralelo
        print(f"\n[PARALLEL] Evaluando {len(neighbors)} vecinos en paralelo con {MAX_THREADS} threads...")
        start_time = time.time()
        results = evaluate_neighbors_parallel(neighbors, iteration, MAX_THREADS)
        elapsed_time = time.time() - start_time
        print(f"[PARALLEL] Evaluación completada en {elapsed_time:.2f} segundos")
        
        # Analizar resultados
        valid_neighbors = []
        valid_objs = []
        failed_neighbors = []
        
        for result in results:
            if result['objective'] < float('inf'):
                valid_neighbors.append(result['config'])
                valid_objs.append(result['objective'])
                safe_print(f"  Vecino {result['id']}: Objetivo = {result['objective']:.6f}")
            else:
                failed_neighbors.append(result)
                safe_print(f"  Vecino {result['id']}: FALLÓ - {result['error']}")
        
        # Resumen de fallos
        print(f"\n[SUMMARY] Vecinos evaluados: {len(results)}")
        print(f"[SUMMARY] Vecinos válidos: {len(valid_neighbors)}")
        print(f"[SUMMARY] Vecinos fallidos: {len(failed_neighbors)}")

        # Eliminar carpetas de esta iteración excepto si es la mejor
        cleanup_iteration_dirs(iteration)
        
        if failed_neighbors:
            print("\n[FAILURES] Análisis de fallos:")
            failure_reasons = {}
            for failed in failed_neighbors:
                reason = failed['error'] or "Unknown error"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            for reason, count in failure_reasons.items():
                print(f"  - {reason}: {count} vecino(s)")
        
        if not valid_neighbors:
            print("[WARNING] No se generaron vecinos válidos")
            break
        
        # Seleccionar vecino
        selected_neighbor, selected_obj = select_neighbor(
            valid_neighbors, valid_objs, current_obj, current_temp)
        
        if selected_neighbor is not None:
            current_config = selected_neighbor
            current_obj = selected_obj
            
            # Actualizar mejor solución
            if current_obj < best_obj:
                best_config = current_config
                best_obj = current_obj
                print(f"\n[NEW BEST] Nueva mejor solución: {best_obj:.6f}")
        
        # Enfriar temperatura
        current_temp *= ALPHA
        
        # Guardar historia
        history.append({
            'iteration': iteration,
            'temperature': current_temp,
            'objective': current_obj,
            'best_objective': best_obj,
            'valid_neighbors': len(valid_neighbors),
            'failed_neighbors': len(failed_neighbors)
        })
        
        print(f"\n[SUMMARY] Iter={iteration}, T={current_temp:.4f}, "
              f"Current={current_obj:.6f}, Best={best_obj:.6f}")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("OPTIMIZACIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"Iteraciones ejecutadas: {iteration}")
    print(f"Temperatura final: {current_temp:.6f}")
    print(f"Mejor objetivo encontrado: {best_obj:.6f}")
    print(f"\nMejor configuración:")
    for key, value in best_config.items():
        if best_config[key] != initial_config[key]:
            print(f"  {key}: {initial_config[key]} → {value}")
    
    return best_config, best_obj, history

def main():
    # Calcular temperatura inicial manualmente corriendo calc_init_temp.py
    INITIAL_TEMP = 175.915313
    
    print(f"[INIT] Usando temperatura inicial: {INITIAL_TEMP}")
    print("[NOTA] Si tienes calc_init_temp.py, ejecútalo primero para calcular T0 óptimo")
    
    # Usar configuración default como solución inicial
    initial_config = DEFAULT_CONFIG.copy()
    
    print("\n[CONFIG] Configuración inicial:")
    for key, value in initial_config.items():
        print(f"  {key}: {value}")
    
    try:
        # Ejecutar Simulated Annealing
        best_config, best_obj, history = simulated_annealing(initial_config, INITIAL_TEMP)
        
        print("\n[SAVING] Guardando resultados en sa_results.txt...")

        if best_obj == float('inf'):
            print("\n[WARN] No hubo solución válida. NO se generará sa_results.txt.")
            return
        
        # Guardar resultados
        with open('sa_results.txt', 'w') as f:
            f.write("SIMULATED ANNEALING RESULTS (PARALLEL VERSION)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best Objective: {best_obj:.6f}\n\n")
            f.write("Best Configuration:\n")
            for key, value in best_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nHistory:\n")
            for h in history:
                valid = h.get('valid_neighbors', 'N/A')
                failed = h.get('failed_neighbors', 'N/A')
                f.write(f"  Iter {h['iteration']}: T={h['temperature']:.4f}, "
                       f"Obj={h['objective']:.6f}, Best={h['best_objective']:.6f}, "
                       f"Valid={valid}, Failed={failed}\n")
        
        print(f"[DONE] Resultados guardados en sa_results.txt")
        print(f"[DONE] Mejor objetivo: {best_obj:.6f}")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ejecución interrumpida por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error en main: {e}")
        import traceback
        print(f"[TRACEBACK] {traceback.format_exc()}")
    finally:
        print("\n[EXIT] Finalizando programa...")

if __name__ == "__main__":
    main()
