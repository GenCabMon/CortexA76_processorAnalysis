#!/bin/bash
# Automatiza la ejecución de McPAT para todas las simulaciones en gem5

# Directorios base
GEM5_DIR="/home/administrador/gem5"
MCPAT_DIR="/home/administrador/mcpat"
SCRIPT_PY="${GEM5_DIR}/scripts/McPAT/gem5toMcPAT_cortexA76.py"
XML_BASE="${GEM5_DIR}/scripts/McPAT/ARM_A76_2.1GHz.xml"
OUTPUT_CSV_DIR="${GEM5_DIR}/results_csv"
CSV_FILE="${OUTPUT_CSV_DIR}/energy_results.csv"

# Crear carpeta de resultados CSV si no existe
mkdir -p "$OUTPUT_CSV_DIR"

# Encabezado del CSV
echo "Name,PeakPower,RuntimeDynamic,TotalLeakage,Energy" > "$CSV_FILE"

# Iterar sobre todas las carpetas m5out*
for sim_dir in ${GEM5_DIR}/m5out*; do
    if [ -d "$sim_dir" ]; then
        sim_name=$(basename "$sim_dir")
        echo "Procesando simulación: $sim_name"

        stats_file="${sim_dir}/stats.txt"
        config_file="${sim_dir}/config.json"
        output_xml="${sim_dir}/config.xml"

        # Verificar que existan los archivos necesarios
        if [ ! -f "$stats_file" ] || [ ! -f "$config_file" ]; then
            echo "  Archivos faltantes en $sim_dir, se omite."
            continue
        fi

        # Generar XML para McPAT
        python3 "$SCRIPT_PY" "$stats_file" "$config_file" "$XML_BASE"
        if [ ! -f "config.xml" ]; then
            echo "  Error al generar config.xml, se omite."
            continue
        fi

        # Mover el XML al directorio correspondiente
        mv config.xml "$output_xml"

        # Ejecutar McPAT
        cd "$MCPAT_DIR" || exit
        ./mcpat -infile "$output_xml" -print_level 1 > "${sim_dir}/mcpat_output.txt"

        # Extraer datos relevantes del archivo mcpat_output.txt
        peak_power=$(grep "Peak Power" "${sim_dir}/mcpat_output.txt" | awk '{print $4}')
        runtime_dynamic=$(grep "Runtime Dynamic" "${sim_dir}/mcpat_output.txt" | head -n 1 | awk '{print $3}')
        total_leakage=$(grep "Total Leakage" "${sim_dir}/mcpat_output.txt" | head -n 1 | awk '{print $4}')

        # Calcular energía aproximada (Energy = PeakPower * SimSeconds)
        sim_seconds=$(grep "simSeconds" "$stats_file" | awk '{print $2}')
        energy=$(echo "$peak_power * $sim_seconds" | bc -l 2>/dev/null)

        # Verificar que los valores sean válidos antes de escribir
        if [[ -z "$peak_power" || -z "$runtime_dynamic" || -z "$total_leakage" || -z "$energy" ]]; then
            echo "  Error al extraer datos de potencia, se omite."
            continue
        fi

        # Guardar en CSV
        echo "${sim_name},${peak_power},${runtime_dynamic},${total_leakage},${energy}" >> "$CSV_FILE"
        echo "  Datos guardados en CSV"
    fi
done

echo "Ejecución completa. Resultados en ${CSV_FILE}"
