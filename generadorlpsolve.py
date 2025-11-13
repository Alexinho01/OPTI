#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convierte instancias (demanda/disposicion) a un modelo .lp
Estructura esperada (salida del generador factible):
instancias_factibles/
  pequena_1_parametros.csv
  pequena_1_demanda.csv  
  pequena_1_disposicion.csv
  ...
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("instancias_factibles")  # carpeta donde están los archivos CSV

def load_instance(instancia_id):
    """Carga una instancia por su ID (ej: 'pequena_1')"""
    
    # Cargar parámetros
    df_params = pd.read_csv(ROOT / f"{instancia_id}_parametros.csv")
    params_dict = dict(zip(df_params['parametro'], df_params['valor']))
    
    # Extraer parámetros
    num_trabajadores = int(params_dict['num_trabajadores'])
    num_dias = int(params_dict['num_dias'])
    num_fines_semana = int(params_dict['num_fines_semana'])
    nivel_tension = params_dict['nivel_tension']
    
    # Extraer turnos
    turnos = []
    i = 1
    while f'turno_{i}' in params_dict:
        turnos.append(params_dict[f'turno_{i}'])
        i += 1
    
    # Extraer fines de semana
    fines_semana = []
    i = 1
    while f'fin_semana_{i}' in params_dict:
        dias_fin_semana = [int(x) for x in params_dict[f'fin_semana_{i}'].split(',')]
        fines_semana.append(dias_fin_semana)
        i += 1
    
    # Cargar disposición
    df_disp = pd.read_csv(ROOT / f"{instancia_id}_disposicion.csv")
    
    # Cargar demanda
    df_dem = pd.read_csv(ROOT / f"{instancia_id}_demanda.csv")
    
    # Crear meta información compatible
    meta = {
        "id": instancia_id,
        "tamaño": params_dict.get('tamaño', ''),
        "num_trabajadores": num_trabajadores,
        "num_dias": num_dias,
        "turnos": turnos,
        "num_fines_semana": num_fines_semana,
        "nivel_tension": nivel_tension,
        "H": num_dias  # para compatibilidad
    }
    
    return meta, turnos, num_dias, df_dem, df_disp, fines_semana

def build_lp_for_instance(meta, T, H, df_dem, df_disp, fines_semana):
    """Construye el modelo .lp a partir de los DataFrames"""
    
    # Conjuntos
    I = meta["num_trabajadores"]  # número de trabajadores
    dias = list(range(1, H+1))
    turnos = T

    # Mapas rápidos: p[i,d,t] (disposición), R[d,t] (demanda)
    p_val = {}
    for _, row in df_disp.iterrows():
        i = int(row["trabajador"].replace("T", ""))  # Convierte "T001" → 1
        d = int(row["dia"])
        t = row["turno"]
        p_val[(i,d,t)] = int(row["disposicion"])

    # Demanda R[d,t]
    R = {}
    for _, row in df_dem.iterrows():
        d = int(row["dia"])
        t = row["turno"]
        R[(d,t)] = int(row["demanda"])

    # Variables x_{i,d,t} - solo si disposición > 0
    def x_name(i,d,t): 
        # Normalizar nombres de variables (sin espacios ni tildes)
        t_clean = t.replace("ñ", "n").replace("á", "a").replace("é", "e")
        return f"x_{i}_{d}_{t_clean}"
    
    X_vars = []
    for i in range(1, I+1):
        for d in dias:
            for t in turnos:
                if p_val.get((i,d,t), 0) > 0:  # Solo si el trabajador tiene disposición > 0
                    X_vars.append((i,d,t))

    # Función objetivo: max sum p * x
    obj_terms = []
    for (i,d,t) in X_vars:
        coef = p_val.get((i,d,t), 0)
        if coef > 0:  # Solo incluir si tiene disposición positiva
            obj_terms.append(f"{coef} {x_name(i,d,t)}")
    
    objetivo = "max: " + (" + ".join(obj_terms) if obj_terms else "0") + ";"

    restricciones = []

    # RESTRICCIÓN MODIFICADA: Cobertura EXACTA de demanda
    # sum_i x_{i,d,t} = R[d,t] para cada (d,t)
    for d in dias:
        for t in turnos:
            rhs = R.get((d,t), 0)
            lhs_vars = [x_name(i,d,t) for (i,d2,t2) in X_vars if d2==d and t2==t]
            if lhs_vars:
                restricciones.append(f"{' + '.join(lhs_vars)} = {rhs};")
            else:
                # Si no hay trabajadores disponibles pero se requiere demanda > 0 → infactible
                if rhs > 0:
                    restricciones.append(f"0 = {rhs}; /* INFACTIBLE: No hay trabajadores para cubrir demanda */")

    # Restricción de máximo 2 turnos por día por trabajador
    for i in range(1, I+1):
        for d in dias:
            diarios = [x_name(i,d,t) for t in turnos if (i,d,t) in X_vars]
            if diarios:
                restricciones.append(f"{' + '.join(diarios)} <= 2;")

    # Restricción de fatiga: Noche(d) + Mañana(d+1) <= 1
    turno_noche = next((t for t in turnos if "noche" in t.lower()), None)
    turno_manana = next((t for t in turnos if "mañana" in t.lower() or "manana" in t.lower()), None)
    
    if turno_noche and turno_manana:
        for i in range(1, I+1):
            for d in dias:
                d_next = d + 1
                if d_next > H:
                    continue
                pair = []
                if (i,d,turno_noche) in X_vars:
                    pair.append(x_name(i,d,turno_noche))
                if (i,d_next,turno_manana) in X_vars:
                    pair.append(x_name(i,d_next,turno_manana))
                if len(pair) == 2:
                    restricciones.append(f"{pair[0]} + {pair[1]} <= 1;")

    # Restricciones de fines de semana (equidad)
    W = len(fines_semana)  # número de fines de semana
    
    def y_name(i,w): return f"y_{i}_{w}"
    Y_vars = []
    for i in range(1, I+1):
        for w in range(1, W+1):
            Y_vars.append((i,w))

    # Variables auxiliares y_{i,w} para indicar si trabajó el fin de semana w
    for i in range(1, I+1):
        for w in range(1, W+1):
            weekend_days = fines_semana[w-1]  # Los días de este fin de semana
            
            # y >= x para cada (d,t) en el fin de semana (si trabaja algún día → y=1)
            for d in weekend_days:
                for t in turnos:
                    if (i,d,t) in X_vars:
                        restricciones.append(f"{x_name(i,d,t)} - {y_name(i,w)} <= 0;")
            
            # y <= sum x (si no trabaja ningún día, y=0)
            sum_weekend = [x_name(i,d,t) for d in weekend_days for t in turnos if (i,d,t) in X_vars]
            if sum_weekend:
                restricciones.append(f"{y_name(i,w)} - " + " - ".join(sum_weekend) + " <= 0;")
            else:
                restricciones.append(f"{y_name(i,w)} = 0;")

    # No 3 fines de semana seguidos trabajados
    for i in range(1, I+1):
        for w in range(1, W-1):  # w desde 1 hasta W-2
            restricciones.append(f"{y_name(i,w)} + {y_name(i,w+1)} + {y_name(i,w+2)} <= 2;")

    # Declaración de variables binarias
    bin_x = [x_name(i,d,t) for (i,d,t) in X_vars]
    bin_y = [y_name(i,w) for (i,w) in Y_vars]

    # Armar texto .lp
    lines = []
    lines.append("/* Modelo de optimización para planificación de enfermería */")
    lines.append("/* Generado automáticamente desde archivos CSV */")
    lines.append("/* Restricción de cobertura: EXACTA (igualdad) */")
    lines.append("")
    lines.append(objetivo)
    lines.append("")
    lines.append("/* Restricciones */")
    lines.extend(restricciones)
    lines.append("")
    lines.append("/* Variables binarias */")
    if bin_x:
        lines.append("bin " + ", ".join(bin_x) + ";")
    if bin_y:
        lines.append("bin " + ", ".join(bin_y) + ";")

    return "\n".join(lines)

def get_all_instance_ids():
    """Obtiene todos los IDs de instancia de los archivos CSV"""
    instance_ids = set()
    
    for file in ROOT.glob("*_parametros.csv"):
        inst_id = file.name.replace("_parametros.csv", "")
        instance_ids.add(inst_id)
    
    return sorted(list(instance_ids))

def analyze_instance_feasibility(meta, df_dem, df_disp):
    """Analiza la factibilidad de la instancia antes de generar el modelo"""
    num_trabajadores = meta["num_trabajadores"]
    num_dias = meta["num_dias"]
    turnos = meta["turnos"]
    
    # Verificar cobertura por turno-día
    issues = []
    
    for _, row in df_dem.iterrows():
        d = int(row["dia"])
        t = row["turno"]
        demanda = int(row["demanda"])
        
        # Contar trabajadores disponibles para este (d,t)
        disponibles = df_disp[
            (df_disp["dia"] == d) & 
            (df_disp["turno"] == t) & 
            (df_disp["disposicion"] > 0)
        ].shape[0]
        
        if disponibles < demanda:
            issues.append(f"Día {d}, Turno {t}: Demanda={demanda}, Disponibles={disponibles}")
    
    return issues

def main():
    if not ROOT.exists():
        raise SystemExit(f"No se encontró la carpeta {ROOT.resolve()}")
    
    # Crear carpeta para modelos LP si no existe
    lp_dir = ROOT / "modelos_lp"
    lp_dir.mkdir(exist_ok=True)
    
    # Obtener todas las instancias
    instance_ids = get_all_instance_ids()
    
    if not instance_ids:
        print("No se encontraron instancias en la carpeta instancias_factibles/")
        print("Asegúrate de ejecutar primero el generador de instancias factibles")
        return
    
    count = 0
    infactibles = 0
    
    for instancia_id in instance_ids:
        try:
            print(f"Procesando {instancia_id}...")
            meta, T, H, df_dem, df_disp, fines_semana = load_instance(instancia_id)
            
            # Analizar factibilidad antes de generar
            issues = analyze_instance_feasibility(meta, df_dem, df_disp)
            if issues:
                print(f"  ⚠ Posibles problemas de factibilidad:")
                for issue in issues[:3]:  # Mostrar solo los primeros 3 problemas
                    print(f"    - {issue}")
                if len(issues) > 3:
                    print(f"    - ... y {len(issues) - 3} más")
                infactibles += 1
            
            # Generar modelo LP
            lp_text = build_lp_for_instance(meta, T, H, df_dem, df_disp, fines_semana)
            
            # Guardar en carpeta modelos_lp
            out_path = lp_dir / f"{instancia_id}.lp"
            out_path.write_text(lp_text, encoding="utf-8")
            count += 1
            print(f"  ✔ Generado: {out_path}")
            
        except Exception as e:
            print(f"  ✗ Error en {instancia_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ ¡Listo! Modelos .lp generados: {count}")
    print(f"⚠️  Instancias con posibles problemas de factibilidad: {infactibles}")
    print(f"📁 Carpeta de salida: {lp_dir}")
    
    if infactibles > 0:
        print(f"\n💡 Recomendación: Revisa las instancias marcadas con 'INFACTIBLE' en los modelos .lp")

if __name__ == "__main__":
    main()