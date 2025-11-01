#!/bin/bash
# Gridsearch automatizado de gem5 usando los top 3 branch predictors indicados
# Ajusta solamente: BP_MAP, L1I_SIZES, L1D_SIZES, BTB_ENTRIES, PIPE_WIDTHS, L2_SIZES, ALU_UNITS
# Limita a 700 simulaciones como máximo.

GEM5_BIN="./build/ARM/gem5.fast"
CONFIG_SCRIPT="scripts/CortexA76_scripts_gem5/CortexA76.py"
BENCHMARK="workloads/jpeg2k_dec/jpg2k_dec"
BENCHMARK_OPTS="-i workloads/jpeg2k_dec/jpg2kdec_testfile.j2k -o image.pgm"

CSV_DIR="results_csv"
mkdir -p "$CSV_DIR"
CSV_FILE="${CSV_DIR}/gridsearch_results.csv"
echo "SimID,BranchPredictor,BP_Type,L1I,L1D,L2,L3,BTB,ALU,ROB,PipelineWidth,SimSeconds,CPI,IPC,ExitCode" > "$CSV_FILE"

# =========================
# PARAMETRIZACIÓN (solo editar estas)
# =========================

# Top 3 branch predictors solicitados: LTAGE, TAGE_SC_L_64KB, TAGE_SC_L_8KB
# índices basados en tu mapping: LTAGE=1, TAGE_SC_L_64KB=8, TAGE_SC_L_8KB=9
declare -A BP_MAP=( ["LTAGE"]=1 ["TAGE_SC_L_64KB"]=8 ["TAGE_SC_L_8KB"]=9 )

# Máximo 3 valores en cada array (excepto L2_SIZES)
declare -a L1I_SIZES=("32kB" "64kB" "128kB")
declare -a L1D_SIZES=("32kB" "64kB" "128kB")
declare -a L2_SIZES=("128kB" "256kB" "512kB" "1MB")   # este puede tener más valores
declare -a L3_SIZES=("1MB" "2MB" "4MB")               # opcional, se itera pero no se limitó
declare -a BTB_ENTRIES=("4096" "8192" "16384")
declare -a ALU_UNITS=("1" "2" "3")
declare -a ROB_ENTRIES=("128" "192" "256")            # dejo 3 opciones aquí también
declare -a PIPE_WIDTHS=("4" "6" "8")

# =========================
# FIN PARAMETRIZACIÓN
# =========================

sim_id=0
MAX_SIMS=2500

# Función para ejecutar una simulación y escribir CSV (simple)
run_and_record() {
  local bp_name=$1
  local bp_type=$2
  local l1i=$3
  local l1d=$4
  local l2=$5
  local l3=$6
  local btb=$7
  local alu=$8
  local rob=$9
  local width=${10}

  sim_id=$((sim_id + 1))
  if [ "$sim_id" -gt "$MAX_SIMS" ]; then
    echo "[INFO] Límite de simulaciones alcanzado (${MAX_SIMS})"
    exit 0
  fi

  local sim_name="gs_bp${bp_type}_L1I${l1i}_L1D${l1d}_L2${l2}_L3${l3}_BTB${btb}_ALU${alu}_ROB${rob}_W${width}"
  local outdir="m5out_${sim_name}"

  echo "[$sim_id] Ejecutando ${sim_name}"

  mkdir -p "${outdir}"
  local cmd="$GEM5_BIN -d ${outdir} $CONFIG_SCRIPT -c $BENCHMARK -o \"$BENCHMARK_OPTS\" \
    --branch_predictor_type=${bp_type} \
    --l1i_size=${l1i} --l1d_size=${l1d} \
    --l2_size=${l2} --l3_size=${l3} \
    --btb_entries=${btb} --num_fu_intALU=${alu} \
    --rob_entries=${rob} \
    --fetch_width=${width} --decode_width=${width} --issue_width=${width} --commit_width=${width}"

  # Guardar comando
  echo "$cmd" > "${outdir}/command.txt"

  # Ejecutar (redirecciona salida)
  eval $cmd > "${outdir}/output.log" 2>&1
  exit_code=$?

  # Extraer métricas si existe stats.txt
  if [ -f "${outdir}/stats.txt" ]; then
    sim_seconds=$(grep "^simSeconds" "${outdir}/stats.txt" | awk '{print $2}')
    sim_seconds=${sim_seconds:-0}
    cpi=$(grep "^system.cpu.cpi" "${outdir}/stats.txt" | awk '{print $2}')
    cpi=${cpi:-0}
    ipc=$(grep "^system.cpu.ipc" "${outdir}/stats.txt" | awk '{print $2}')
    ipc=${ipc:-0}
  else
    sim_seconds=0
    cpi=0
    ipc=0
  fi

  # Escribir línea CSV
  echo "${sim_id},${bp_name},${bp_type},${l1i},${l1d},${l2},${l3},${btb},${alu},${rob},${width},${sim_seconds},${cpi},${ipc},${exit_code}" >> "$CSV_FILE"
}

# Loop principal: generar combinaciones y ejecutar
for bp_name in "${!BP_MAP[@]}"; do
  bp_type=${BP_MAP[$bp_name]}
  for l1i in "${L1I_SIZES[@]}"; do
    for l1d in "${L1D_SIZES[@]}"; do
      for l2 in "${L2_SIZES[@]}"; do
        for l3 in "${L3_SIZES[@]}"; do
          for btb in "${BTB_ENTRIES[@]}"; do
            for alu in "${ALU_UNITS[@]}"; do
              for rob in "${ROB_ENTRIES[@]}"; do
                for width in "${PIPE_WIDTHS[@]}"; do
                  run_and_record "$bp_name" "$bp_type" "$l1i" "$l1d" "$l2" "$l3" "$btb" "$alu" "$rob" "$width"
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Gridsearch completado. Resultados en: $CSV_FILE"
