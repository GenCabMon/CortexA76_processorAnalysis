#!/bin/bash

# Script de automatizacion para simulaciones gem5 Cortex A76
# Genera un CSV separado por cada grupo de simulaciones

# Configuracion base
GEM5_BIN="./build/ARM/gem5.fast"
CONFIG_SCRIPT="scripts/CortexA76_scripts_gem5/CortexA76.py"
BENCHMARK="workloads/jpeg2k_dec/jpg2k_dec"
BENCHMARK_OPTS="-i workloads/jpeg2k_dec/jpg2kdec_testfile.j2k -o image.pgm"

# Directorio para CSVs
CSV_DIR="results_csv"
mkdir -p $CSV_DIR

# Archivo maestro con todos los resultados
CSV_MASTER="${CSV_DIR}/00_all_results.csv"

# CSV header comun
CSV_HEADER="SimID,Name,Description,Parameters,SimSeconds,HostSeconds,CPI,IPC,BranchMispreds,BTBMispreds,PredictorMispreds,NumInsts,NumOps,ExitCode"

echo "========================================"
echo "  GEM5 ADAPTIVE SIMULATION SUITE"
echo "========================================"
echo ""

# Crear CSV maestro
echo "$CSV_HEADER" > $CSV_MASTER

# Contadores globales
total_count=0
success_count=0
fail_count=0

# Variables para el mejor BP
BEST_BP_TYPE=10
BEST_BP_NAME="Tournament"

# Variable para el CSV del grupo actual
CURRENT_GROUP_CSV=""

# ============================================
# FUNCION: Iniciar nuevo grupo
# ============================================
start_group() {
    local group_num=$1
    local group_name=$2
    CURRENT_GROUP_CSV="${CSV_DIR}/${group_num}_${group_name}.csv"
    echo "$CSV_HEADER" > "$CURRENT_GROUP_CSV"
    echo ""
    echo "========================================"
    echo "  GRUPO ${group_num}: ${group_name}"
    echo "========================================"
    echo ""
}

# ============================================
# FUNCION: Finalizar grupo
# ============================================
end_group() {
    local group_name=$1
    echo ""
    echo "[OK] Grupo completado: ${group_name}"
    echo "[INFO] CSV guardado: ${CURRENT_GROUP_CSV}"
    echo ""
    
    # Mostrar top 3 del grupo
    local num_lines=$(wc -l < "$CURRENT_GROUP_CSV")
    if [ $num_lines -gt 1 ]; then
        echo "[TOP 3] Mejores de este grupo (por SimSeconds):"
        tail -n +2 "$CURRENT_GROUP_CSV" | sort -t',' -k5 -n | head -3 | \
            awk -F',' '{printf "   %d. %s: %.6fs (CPI=%.4f)\n", NR, $2, $5, $7}'
        echo ""
    fi
}

# ============================================
# FUNCION: Comparar flotantes
# ============================================
float_lt() {
    awk -v n1="$1" -v n2="$2" 'BEGIN {if (n1<n2) exit 0; exit 1}'
}

# ============================================
# FUNCION: Ejecutar simulacion
# ============================================
run_simulation() {
    local sim_id=$1
    local sim_name=$2
    local sim_params=$3
    local sim_desc=$4
    local outdir="m5out_${sim_name}"
    
    total_count=$((total_count + 1))
    
    echo "----------------------------------------"
    echo "[${total_count}] Simulacion: ${sim_name}"
    echo "----------------------------------------"
    echo "Descripcion: ${sim_desc}"
    echo "Parametros: ${sim_params:-ninguno}"
    
    # Crear directorio
    mkdir -p ${outdir}
    
    # Construir comando
    if [ -z "$sim_params" ]; then
        cmd="$GEM5_BIN -d ${outdir} $CONFIG_SCRIPT -c $BENCHMARK -o \"$BENCHMARK_OPTS\""
    else
        cmd="$GEM5_BIN -d ${outdir} $CONFIG_SCRIPT -c $BENCHMARK -o \"$BENCHMARK_OPTS\" $sim_params"
    fi
    
    echo "$cmd" > "${outdir}/command.txt"
    
    # Ejecutar
    echo "[RUN] Ejecutando..."
    start_time=$(date +%s)
    
    eval $cmd > ${outdir}/output.log 2>&1
    exit_code=$?
    
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    # Extraer metricas
    if [ $exit_code -eq 0 ] && [ -f "${outdir}/stats.txt" ]; then
        success_count=$((success_count + 1))
        echo "[OK] Completado en ${elapsed}s"
        
        # Extraer datos
        sim_seconds=$(grep "^simSeconds" "${outdir}/stats.txt" | head -1 | awk '{print $2}')
        host_seconds=$(grep "^hostSeconds" "${outdir}/stats.txt" | head -1 | awk '{print $2}')
        cpi=$(grep "^system.cpu.cpi " "${outdir}/stats.txt" | head -1 | awk '{print $2}')
        ipc=$(grep "^system.cpu.ipc " "${outdir}/stats.txt" | head -1 | awk '{print $2}')
        
        # Branch mispredictions
        branch_mispreds=$(grep "system.cpu.commit.branchMispredicts" "${outdir}/stats.txt" | head -1 | awk '{print $2}')
        btb_mispreds=$(grep "system.cpu.branchPred.BTBMisses" "${outdir}/stats.txt" | head -1 | awk '{print $2}')
        
        if [ -z "$btb_mispreds" ]; then
            btb_mispreds="0"
        fi
        
        if [ -n "$branch_mispreds" ] && [ -n "$btb_mispreds" ]; then
            predictor_mispreds=$((branch_mispreds - btb_mispreds))
        else
            predictor_mispreds="0"
        fi
        
        # Instructions y Ops
        num_insts=$(grep "system.cpu.commitStats0.numInsts" "${outdir}/stats.txt" | head -1 | awk '{print $2}')
        num_ops=$(grep "system.cpu.commitStats0.numOps" "${outdir}/stats.txt" | head -1 | awk '{print $2}')
        
        # Valores por defecto
        sim_seconds=${sim_seconds:-0}
        host_seconds=${host_seconds:-0}
        cpi=${cpi:-0}
        ipc=${ipc:-0}
        branch_mispreds=${branch_mispreds:-0}
        num_insts=${num_insts:-0}
        num_ops=${num_ops:-0}
        
        echo "[STATS] SimSeconds: ${sim_seconds} | CPI: ${cpi} | IPC: ${ipc}"
        
        # Crear linea CSV
        csv_line="${sim_id},${sim_name},\"${sim_desc}\",\"${sim_params}\",${sim_seconds},${host_seconds},${cpi},${ipc},${branch_mispreds},${btb_mispreds},${predictor_mispreds},${num_insts},${num_ops},${exit_code}"
        
        # Escribir a CSV del grupo actual
        echo "$csv_line" >> "$CURRENT_GROUP_CSV"
        
        # Escribir a CSV maestro
        echo "$csv_line" >> "$CSV_MASTER"
        
        # Retornar SimSeconds
        echo "$sim_seconds"
    else
        fail_count=$((fail_count + 1))
        echo "[ERROR] Fallo (codigo ${exit_code})"
        if [ -f "${outdir}/output.log" ]; then
            echo "[ERROR] Ultimas lineas:"
            tail -10 "${outdir}/output.log"
        fi
        
        csv_line="${sim_id},${sim_name},\"${sim_desc}\",\"${sim_params}\",0,0,0,0,0,0,0,0,0,${exit_code}"
        echo "$csv_line" >> "$CURRENT_GROUP_CSV"
        echo "$csv_line" >> "$CSV_MASTER"
        
        echo "999999"
    fi
    
    echo ""
}

# ============================================
# FASE 1: EVALUAR BRANCH PREDICTORS
# ============================================
echo "========================================"
echo "  FASE 1: EVALUANDO BRANCH PREDICTORS"
echo "  (Se seleccionara el mejor)"
echo "========================================"
echo ""

start_group "01" "branch_predictors"

declare -a BP_NAMES=("BiMode" "LTAGE" "Local" "MPP64KB" "MPP8KB" "MPPTAGE64KB" "MPPTAGE8KB" "TAGE" "TAGE_SC_L_64KB" "TAGE_SC_L_8KB" "Tournament")

min_sim_seconds=999999
best_bp_type=10
best_bp_name="Tournament"

for bp_type in {0..10}; do
    result=$(run_simulation "BP_${bp_type}" "bp_${BP_NAMES[$bp_type]}" \
        "--branch_predictor_type=${bp_type}" \
        "Branch Predictor: ${BP_NAMES[$bp_type]}")
    
    if float_lt "$result" "$min_sim_seconds"; then
        min_sim_seconds=$result
        best_bp_type=$bp_type
        best_bp_name=${BP_NAMES[$bp_type]}
    fi
done

end_group "Branch Predictors"

# Mostrar mejor BP
echo "========================================"
echo "  RESULTADOS BRANCH PREDICTORS"
echo "========================================"
echo ""
echo "[RANKING] Top 5 Branch Predictors:"
echo ""

tail -n +2 "${CSV_DIR}/01_branch_predictors.csv" | sort -t',' -k5 -n | head -5 | while IFS=',' read -r sim_id name desc params sim_sec rest; do
    if [[ "$name" == *"${best_bp_name}"* ]]; then
        echo "   [BEST] ${name}: ${sim_sec}s (SELECCIONADO)"
    else
        echo "   ${name}: ${sim_sec}s"
    fi
done

echo ""
echo "========================================"
echo "  MEJOR BRANCH PREDICTOR SELECCIONADO"
echo "========================================"
echo ""
echo "  Tipo: ${best_bp_type} - ${best_bp_name}"
echo "  SimSeconds: ${min_sim_seconds}s"
echo ""
echo "  Se usara para el resto de simulaciones"
echo ""
echo "========================================"
echo ""

BEST_BP_TYPE=$best_bp_type
BEST_BP_NAME=$best_bp_name

# ============================================
# FASE 2: RESTO DE SIMULACIONES
# ============================================
echo "========================================"
echo "  FASE 2: OTRAS CONFIGURACIONES"
echo "  Usando BP: ${best_bp_name} (tipo ${best_bp_type})"
echo "========================================"
echo ""

# ============================================
# GRUPO 2: L1 INSTRUCTION CACHE
# ============================================
start_group "02" "l1_instruction_cache"

for size in 32kB 64kB 128kB 256kB; do
    run_simulation "L1I_${size}" "l1i_${size}" \
        "--branch_predictor_type=${BEST_BP_TYPE} --l1i_size=${size}" \
        "L1I: ${size} + BP: ${BEST_BP_NAME}"
done

end_group "L1 Instruction Cache"

# ============================================
# GRUPO 3: L1 DATA CACHE
# ============================================
start_group "03" "l1_data_cache"

for size in 32kB 64kB 128kB 256kB; do
    run_simulation "L1D_${size}" "l1d_${size}" \
        "--branch_predictor_type=${BEST_BP_TYPE} --l1d_size=${size}" \
        "L1D: ${size} + BP: ${BEST_BP_NAME}"
done

end_group "L1 Data Cache"

# ============================================
# GRUPO 4: L2 CACHE
# ============================================
start_group "04" "l2_cache"

for size in 128kB 256kB 512kB 1MB; do
    run_simulation "L2_${size}" "l2_${size}" \
        "--branch_predictor_type=${BEST_BP_TYPE} --l2_size=${size}" \
        "L2: ${size} + BP: ${BEST_BP_NAME}"
done

end_group "L2 Cache"

# ============================================
# GRUPO 5: L3 CACHE
# ============================================
start_group "05" "l3_cache"

for size in 1MB 2MB 4MB; do
    run_simulation "L3_${size}" "l3_${size}" \
        "--branch_predictor_type=${BEST_BP_TYPE} --l3_size=${size}" \
        "L3: ${size} + BP: ${BEST_BP_NAME}"
done

end_group "L3 Cache"

# ============================================
# GRUPO 6: BTB ENTRIES
# ============================================
start_group "06" "btb_entries"

for entries in 1024 2048 4096 8192 16384 32768; do
    run_simulation "BTB_${entries}" "btb_${entries}" \
        "--branch_predictor_type=${BEST_BP_TYPE} --btb_entries=${entries}" \
        "BTB: ${entries} + BP: ${BEST_BP_NAME}"
done

end_group "BTB Entries"

# ============================================
# GRUPO 7: INTEGER ALU UNITS
# ============================================
start_group "07" "integer_alu_units"

for units in 1 2 3 4; do
    run_simulation "ALU_${units}" "alu_${units}" \
        "--branch_predictor_type=${BEST_BP_TYPE} --num_fu_intALU=${units}" \
        "ALU: ${units} units + BP: ${BEST_BP_NAME}"
done

end_group "Integer ALU Units"

# ============================================
# GRUPO 8: PIPELINE WIDTHS
# ============================================
start_group "08" "pipeline_widths"

for width in 2 4 6 8; do
    run_simulation "PIPE_${width}" "pipe_${width}" \
        "--branch_predictor_type=${BEST_BP_TYPE} --fetch_width=${width} --decode_width=${width} --rename_width=${width} --commit_width=${width}" \
        "Pipeline: ${width}-wide + BP: ${BEST_BP_NAME}"
done

run_simulation "PIPE_mixed1" "pipe_mixed_narrow" \
    "--branch_predictor_type=${BEST_BP_TYPE} --fetch_width=6 --decode_width=6 --issue_width=8 --commit_width=4" \
    "Pipeline: Wide front + BP: ${BEST_BP_NAME}"

run_simulation "PIPE_mixed2" "pipe_mixed_wide" \
    "--branch_predictor_type=${BEST_BP_TYPE} --fetch_width=4 --decode_width=4 --issue_width=12 --commit_width=6" \
    "Pipeline: Wide back + BP: ${BEST_BP_NAME}"

end_group "Pipeline Widths"

# ============================================
# GRUPO 9: QUEUE SIZES
# ============================================
start_group "09" "queue_sizes"

run_simulation "ROB_64" "rob_64" \
    "--branch_predictor_type=${BEST_BP_TYPE} --rob_entries=64" \
    "ROB: 64 + BP: ${BEST_BP_NAME}"

run_simulation "ROB_256" "rob_256" \
    "--branch_predictor_type=${BEST_BP_TYPE} --rob_entries=256" \
    "ROB: 256 + BP: ${BEST_BP_NAME}"

run_simulation "LSQ_small" "lsq_small" \
    "--branch_predictor_type=${BEST_BP_TYPE} --lq_entries=32 --sq_entries=32" \
    "LSQ: 32/32 + BP: ${BEST_BP_NAME}"

run_simulation "LSQ_large" "lsq_large" \
    "--branch_predictor_type=${BEST_BP_TYPE} --lq_entries=128 --sq_entries=128" \
    "LSQ: 128/128 + BP: ${BEST_BP_NAME}"

end_group "Queue Sizes"

# ============================================
# GRUPO 10: OPTIMIZACIONES
# ============================================
start_group "10" "optimizations"

run_simulation "OPT_01" "opt_best_cache" \
    "--branch_predictor_type=${BEST_BP_TYPE} --l1i_size=128kB --l1d_size=128kB --l2_size=512kB" \
    "Large Caches + Best BP"

run_simulation "OPT_02" "opt_wide_pipeline" \
    "--branch_predictor_type=${BEST_BP_TYPE} --fetch_width=6 --decode_width=6 --issue_width=10 --commit_width=6 --rob_entries=192" \
    "Wide Pipeline + Best BP"

run_simulation "OPT_03" "opt_memory_hierarchy" \
    "--branch_predictor_type=${BEST_BP_TYPE} --l1i_size=128kB --l1d_size=128kB --l2_size=1MB --l3_size=4MB" \
    "Large Memory + Best BP"

run_simulation "OPT_04" "opt_execution_units" \
    "--branch_predictor_type=${BEST_BP_TYPE} --num_fu_intALU=3 --num_fu_FP_SIMD_ALU=3 --num_fu_read=3 --num_fu_write=2" \
    "More Exec Units + Best BP"

run_simulation "OPT_05" "opt_branch_hardware" \
    "--branch_predictor_type=${BEST_BP_TYPE} --btb_entries=16384 --ras_entries=32" \
    "Enhanced Branch HW + Best BP"

run_simulation "OPT_06" "opt_low_latency" \
    "--branch_predictor_type=${BEST_BP_TYPE} --l1i_lat=1 --l1d_lat=2 --l2_lat=6" \
    "Low Latency + Best BP"

run_simulation "OPT_07" "opt_balanced" \
    "--branch_predictor_type=${BEST_BP_TYPE} --l1i_size=128kB --l1d_size=128kB --num_fu_intALU=3 --btb_entries=8192" \
    "Balanced Config + Best BP"

run_simulation "OPT_08" "opt_aggressive" \
    "--branch_predictor_type=${BEST_BP_TYPE} --l1i_size=256kB --l1d_size=256kB --l2_size=1MB --fetch_width=8 --issue_width=12" \
    "Aggressive Perf + Best BP"

run_simulation "OPT_09" "opt_efficiency" \
    "--branch_predictor_type=${BEST_BP_TYPE} --l1i_size=32kB --l1d_size=32kB --fetch_width=2 --issue_width=4" \
    "Efficiency + Best BP"

run_simulation "OPT_10" "opt_ultimate" \
    "--branch_predictor_type=${BEST_BP_TYPE} --l1i_size=256kB --l1d_size=256kB --l2_size=1MB --l3_size=4MB --btb_entries=32768 --num_fu_intALU=4 --fetch_width=8 --issue_width=12 --rob_entries=256" \
    "Ultimate Performance"

end_group "Optimizations"

# ============================================
# RESUMEN FINAL
# ============================================
echo "========================================"
echo "  SIMULACIONES COMPLETADAS"
echo "========================================"
echo ""
echo "[OK] Exitosas: ${success_count}/${total_count}"
echo "[FAIL] Fallidas: ${fail_count}/${total_count}"
echo ""

echo "Archivos CSV generados en ${CSV_DIR}/:"
ls -1 ${CSV_DIR}/*.csv | while read file; do
    filename=$(basename "$file")
    line_count=$(($(wc -l < "$file") - 1))
    echo "   ${filename} (${line_count} simulaciones)"
done
echo ""

# Top 5 global
echo "[TOP 5] MEJOR RENDIMIENTO (menor SimSeconds):"
tail -n +2 $CSV_MASTER | sort -t',' -k5 -n | head -5 | \
    awk -F',' '{printf "   %d. %s: %.6fs (CPI=%.4f)\n", NR, $2, $5, $7}'

echo ""
echo "[TOP 5] MEJOR CPI:"
tail -n +2 $CSV_MASTER | sort -t',' -k7 -n | head -5 | \
    awk -F',' '{printf "   %d. %s: CPI=%.4f (%.6fs)\n", NR, $2, $7, $5}'

echo ""
echo "[INFO] Branch Predictor usado: ${BEST_BP_NAME} (tipo ${BEST_BP_TYPE})"
echo "[INFO] Usa analyze_results.py para visualizar"
echo ""
